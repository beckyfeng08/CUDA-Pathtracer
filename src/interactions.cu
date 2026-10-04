#include "interactions.h"
#include "intersections.h"

#include "utilities.h"
#include <glm/gtc/constants.hpp>
#include <thrust/random.h>

// -------- DIFFUSE RELATED FUNCTIONS --------- //
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 sampleDiffuse(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;
    return m.color;
}


// -------- DIELECTRIC RELATED FUNCTIONS --------- //
__host__ __device__ glm::vec3 samplePerfectSpecularReflection(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m) 
{
    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;
    return m.color;
}

__host__ __device__ glm::vec3 samplePerfectSpecularTransmission(
    float eta,
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m) 
{

    glm::vec3 refraction_dir = glm::refract(pathSegment.ray.direction, normal, eta);

    pathSegment.ray.direction = glm::normalize(refraction_dir);
    pathSegment.ray.origin =
        intersect + pathSegment.ray.direction * EPSILON *  100.f;

    return m.color;
}

// fresnel reflection from dielectrics and unpolarized light
__host__ __device__ float computeFresnelReflectance(float cosThetaI, float eta)
{
    
    float sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI * eta * eta;

    float F;

    if (sin2ThetaT >= 1.0f) F = 1.0f; // total internal reflection
    else
    {
        float cosThetaT = glm::sqrt(1.0f - sin2ThetaT);

        float rPerp =
            (eta * cosThetaI -  cosThetaT) /
            (eta * cosThetaI + cosThetaT);

        float rPar =
            (cosThetaI - eta * cosThetaT) /
            (cosThetaI + eta * cosThetaT);

        F = 0.5f * (rPerp * rPerp + rPar * rPar);
    }
    return F;
}

// another way to calculate fresnel, faster but less physically accurate
__host__ __device__ float computeSchlickApproxF(float cosThetaI, float eta)
{
    float R0 = ((eta - 1.f) * (eta - 1.f) / ((eta + 1.f) * (eta + 1.f)));
    return R0 + (1.f - R0) * glm::pow(1.f - cosThetaI, 5.f);
}

__host__ __device__ glm::vec3 sampleDielectric(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 resulting_color = m.color;
    
    float cosThetaI = glm::dot(normal, -pathSegment.ray.direction);
    float etaI = 1.0f;
    float etaT = m.indexOfRefraction;

    if (cosThetaI < 0.0f)
    {
        // exiting the material
        cosThetaI = -cosThetaI;
        etaI = m.indexOfRefraction;
        etaT = 1.0f;
    }

    float eta = etaI / etaT;

    //float F = computeFresnelReflectance(cosThetaI, eta);
    float F = computeSchlickApproxF(cosThetaI, eta);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float probability = u01(rng);

    // russion roulette choose
    if (probability < F) {
        resulting_color = samplePerfectSpecularReflection(
            pathSegment,
            intersect,
            normal,
            m);
    } else {
        resulting_color = samplePerfectSpecularTransmission(
            eta,
            pathSegment,
            intersect,
            normal,
            m);
    }

    return resulting_color;
}

// TODO: clean up any vars not used in the code?
__host__ __device__ glm::vec3 sampleDirectLighting(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    Geom* geoms,
    int geoms_size,
    Light* l,
    int lights_size,
    thrust::default_random_engine& rng
)
{
    glm::vec3 resulting_color = glm::vec3(1.f);

    thrust::uniform_real_distribution<float> u01(0, 1);
    int light_index = floor(u01(rng) * lights_size);
    Light light = l[light_index];

    if (light.type == AREALIGHT) 
    {
        glm::vec3 xy = glm::vec3(glm::mix(-0.5, 0.5, u01(rng)), 0.0,glm::mix(-0.5, 0.5, u01(rng))); // randomly sample a point in the arealight

        glm::vec3 rand_pt_light_w = glm::vec3(light.transform * glm::vec4(xy, 1.)); // take random point in local light, transform to world coordinate

        glm::vec3 view_point = pathSegment.ray.origin;
        glm::vec3 wiW = glm::normalize(rand_pt_light_w - view_point);

        float cosTheta = glm::dot(light.normal, -wiW);
        // we are behind the light or our surface is facing away from the light
        if (cosTheta <= 0. ) {
            resulting_color = glm::vec3(0.f);
        }
        else
        {
            // check for occluders
            Ray ray = {view_point, wiW};
            int hit_geom_index = -1;

            // throwaway vars
            glm::vec3 normal;
            glm::vec3 intersectionPoint;
            bool outside = false;
            float t = geometryIntersectionTest(
                geoms,
                geoms_size,
                ray,
                intersectionPoint,
                normal,
                outside,
                hit_geom_index
            ); 

            float r = glm::length(rand_pt_light_w - view_point);

            // we hit an occluder before reaching the light
            if (hit_geom_index != -1 && t < r)
            {
                resulting_color = glm::vec3(0.);

            } 
            else 
            {
                float area = light.scale.x * light.scale.z; // area of arealight

                float pdf_dA = 1.f / area;
                float pdf = pdf_dA * r*r / cosTheta; // account for falloff, and angle
                resulting_color = light.color * light.intensity / pdf;

            }
        }
    }
    else if (light.type == POINTLIGHT) 
    {
        glm::vec3 view_point = pathSegment.ray.origin;

        glm::vec3 wiW = glm::normalize(light.translation - view_point);
        float r = glm::length(light.translation - view_point);
        float ndotl = glm::dot(normal, wiW);
        //  our surface is facing away from the light
        if (ndotl <= 0. || r > light.pointLight.range) {
            resulting_color = glm::vec3(0.);
        }
        else
        {

            Ray ray = {view_point + normal * EPSILON, wiW}; // prevent self intersections
            // throwaway vars
            int hit_geom_index = -1;
            glm::vec3 normal;
            glm::vec3 intersectionPoint;
            bool outside = false;
            float t = geometryIntersectionTest(
                    geoms,
                    geoms_size,
                    ray,
                    intersectionPoint,
                    normal,
                    outside,
                    hit_geom_index
                );


            if (hit_geom_index != -1 && t < r) // hit an occluder
                resulting_color = glm::vec3(0.);
            else {
                resulting_color = light.color * light.intensity * ndotl / (r * r) ;
            }
        }
    }

    return resulting_color;
}


__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    Geom* geoms,
    int geoms_size,
    Light* l,
    int lights_size,
    thrust::default_random_engine& rng)
{
    glm::vec3 resulting_color;

    if (m.isDielectric > 0.)
    {
        resulting_color = sampleDielectric(
            pathSegment,
            intersect,
            normal,
            m,
            rng);

        // Indirect
        pathSegment.color *= resulting_color;

    }
    else
    {
        resulting_color = sampleDiffuse(
            pathSegment,
            intersect,
            normal,
            m,
            rng);

        glm::vec3 resulting_color_direct = sampleDirectLighting(
            pathSegment,
            intersect,
            normal,
            geoms,
            geoms_size,
            l,
            lights_size,
            rng
        );
        // Direct lighting
        float pdf = PI;

        // Indirect
        pathSegment.color *= resulting_color;
        pathSegment.radiance += pathSegment.color * resulting_color_direct / pdf;
    }

    if (pathSegment.color == glm::vec3(0.f))
        pathSegment.remainingBounces = 0;
}