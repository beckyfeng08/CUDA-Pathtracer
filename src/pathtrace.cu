#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static Light* dev_lights = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need


    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_lights);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // jitter the ray for antialiasing effects
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float offsetx = u01(rng) - 0.5; // -0.5 to 0.5
        float offsety =  u01(rng) - 0.5; // -0.5 to 0.5

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + offsetx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + offsety - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Light* lights,
    int lights_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        int hit_light_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        // parse through lights
        for (int i = 0; i < lights_size; i++)
        {
            Light& light = lights[i];
            if (light.type == AREALIGHT)
            {
                // use a cube as a dummy
                Geom placeholdersquare = {
                    CUBE, 
                    0, 
                    light.translation,
                    light.rotation, 
                    glm::vec3(light.areaLight.x * light.scale.x, EPSILON * 10.f, light.areaLight.y * light.scale.y),
                    light.transform, 
                    light.inverseTransform, 
                    light.invTranspose
                };

                t = boxIntersectionTest(placeholdersquare, pathSegment.ray, tmp_intersect, tmp_normal, outside);

            } 
            //else if (light.type == POINTLIGHT)
            //{
            //    // if light is point light, itll miss. Be sure to use direct lighting for this to render
            //    
            //}

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_light_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1 && hit_light_index == -1) // no geometry was hit
        {
            intersections[path_index].t = -1.0f;
        }
        else if (hit_light_index != -1) // if we hit a light, then this should be updated, and be the first object the ray hits (updated t_min)
        {
            intersections[path_index].t = t_min;
            intersections[path_index].isLight = 1;
            intersections[path_index].lightId = hit_light_index;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].materialId = -1; // no material id
        } 
        else if (hit_geom_index != -1 ) 
        {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].isLight = 0;

        }
    }
}


__global__ void shadeMaterial(
    int iter,int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Light* lights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];

    // intersection doesn't hit anything or behind camera
    if (intersection.t <= 0.0f) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];
    Light light = lights[intersection.lightId];

    // If we actually hit a light, light the ray
    if (intersection.isLight) {
        // TODO: handle light contributions depending on area or point light
        pathSegment.color *= light.color * light.intensity;
        pathSegment.remainingBounces = 0;
        return;
    }
  
    // If the material indicates that the object was a light, "light" the ray
    if (material.emittance > 0.0f) {
        pathSegment.color *= (material.color * material.emittance);
        pathSegment.remainingBounces = 0;
        return;
    }
    
    if (pathSegment.remainingBounces <= 0) { // no contributionn if no more bounces
        pathSegment.color = glm::vec3(0.f);
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

    glm::vec3 intersectPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

    // spawn new ray
    scatterRay(
        pathSegment,
        intersectPoint,
        intersection.surfaceNormal,
        material,
        rng
    );

    // after this, then our pathSegment should be completely updated here for the ray

    // direct lightin
    // sampleDirectLighting(pathSegment,
    //     intersectPoint,
    //     intersection.surfaceNormal,
    //     material,
    //     lights,
    //     rng);
   
    pathSegment.remainingBounces--;


}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces <= 0) 
            image[iterationPath.pixelIndex] += iterationPath.color;
    }
}
// helper for thrust::sort. Sorts the array based on materialID
struct sort_by_material {
    __host__ __device__ bool operator()(const thrust::tuple<ShadeableIntersection, PathSegment>& zipped_a, const thrust::tuple<ShadeableIntersection, PathSegment>& zipped_b) const
    {
        return thrust::get<0>(zipped_a).materialId < thrust::get<0>(zipped_b).materialId;// terminate if we don't intersect anything and if we are out of bounces
    }
};

//helper for thrust::removeif. Checks to see if intersection < 0. If so, terminate
struct terminateRays {
    __host__ __device__ bool operator()(const PathSegment& pathsegment) const
    {
        return pathsegment.remainingBounces <= 0;// terminate if we don't intersect anything and if we are out of bounces
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;


    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths); // set ray origin and ray dir, in dev_paths (pathsegments)
    // multiple iterations?

    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount; // end of the array
    int num_paths = dev_path_end - dev_paths; // number of pathsegments

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lights,
            hst_scene->lights.size(),
            dev_intersections
        );
        // dev_intersections should now be populated
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // zip up with dev_paths, so the indices match
        auto dev_zipped = thrust::make_zip_iterator(thrust::make_tuple(dev_intersections, dev_paths));
        auto dev_zipped_end = thrust::make_zip_iterator(thrust::make_tuple(dev_intersections + num_paths, dev_paths + num_paths));

        // making contiguous in memory, sort by materialID
        thrust::sort(thrust::device, dev_zipped, dev_zipped_end, sort_by_material());

        // apply bsdf and populate color of paths
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_lights
        );


        checkCUDAError("Shading material");
   

        // Stream compact away rays that don't intersect
        
        // copy color data to the image before we completely terminate the rays
        finalGather << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_image, dev_paths);

        // terminate rays that have no more bounces
        dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_path_end, terminateRays());
        num_paths = dev_path_end - dev_paths;
        
        if (num_paths == 0)
            iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
