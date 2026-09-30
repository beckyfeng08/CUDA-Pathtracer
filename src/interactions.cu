#include "interactions.h"

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
__host__ __device__ float computeFresnelReflectance(float cosThetaI, float etaI, float etaT)
{

    if (cosThetaI < 0.0f)
    {
        // exiting the material
        cosThetaI = -cosThetaI;
        etaI = m.indexOfRefraction;
        etaT = 1.0f;
    }

    float eta = etaI / etaT;
    cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);
    
    float sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI * eta * eta;

    float F;

    if (sin2ThetaT >= 1.0f) F = 1.0f; // total internal reflection
    else
    {
        float cosThetaT = glm::sqrt(1.0f - sin2ThetaT);

        float rPerp =
            (etaI * cosThetaI - etaT * cosThetaT) /
            (etaI * cosThetaI + etaT * cosThetaT);

        float rPar =
            (etaT * cosThetaI - etaI * cosThetaT) /
            (etaT * cosThetaI + etaI * cosThetaT);

        F = 0.5f * (rPerp * rPerp + rPar * rPar);
    }
    return F;
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
    float F = computeFresnelReflectance(cosThetaI, etaI, etaT);

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

__host__ __device__ glm::vec3 sampleDirectLighting(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    Lights* l,
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
        glm::vec3 xy = glm::vec3(glm::mix(-0.5, 0.5, u01(rng)), glm::mix(-0.5, 0.5, u01(rng)), 0.0); // randomly sample a point in the arealight

        glm::vec3 rand_pt_light_w = glm::vec3(light.transform.T * glm::vec4(xy, 1.)); // take random point in local light, transform to world coordinate

        glm::vec3 view_point = pathSegment.origin;
        glm::vec3 wiW = glm::normalize(rand_pt_light_w - view_point);

        float cosTheta = glm::dot(light.normal, -wiW);
        if (cosTheta <= 0.) { // we are behind the light
            resulting_color = glm::vec3(0.);
        } 
        else // we are on the side of the light that it is facing
        {
            float area = light.scale.x * light.scale.z; // area of arealight
            float r = glm::length(wiW);

            float pdf_dA = 1.f / area;
            pdf = pdf_dA * r*r / cosTheta; // account for falloff, and angle

            // how to do?? need to call computeIntersections?? but we are on the gpu already
            // put geometry intersections on the device, then call
            
            PathSegment p = pathSegment;

            Ray ray = SpawnRay(view_point, wiW);
            Intersection isect = sceneIntersect(ray); // problem here??? how to make since our intersection test is gpu acc

            if (isect.t == -1 && isect.t < r - 1e-3) { // hit an occluder
                resulting_color = glm::vec3(0.);
            } else {
                resulting_color = light.color * light.intensity / pdf;
            }
        }
    }
    else if (light.type == POINTLIGHT) 
    {

    }

    return resulting_color;
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    Lights* l,
    int lights_size,
    thrust::default_random_engine &rng)
{

    glm::vec3 resulting_color;

    if (m.isDielectric > 0.) { // glass, water, plastic
        resulting_color = sampleDielectric(
            pathSegment,
            intersect,
            normal,
            m,
            rng);
    }
    // DIFFUSE
    else {
        resulting_color = sampleDiffuse(
            pathSegment,
            intersect,
            normal,
            m,
            rng); 
    }

    resulting_color *= sampleDirectLighting(
        pathSegment,
        intersect,
        normal,
        m, 
        l,
        lights_size,
        rng
        );
    pathSegment.color *= resulting_color;

}
