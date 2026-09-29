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
__host__ __device__ float computeFresnelReflectance(float costhetaI, float n_i, float n_t)
{
    // swap indices of refraction if needed
    bool entering = costhetaI > 0.f;
    if (!entering) {
        float temp = n_i;
        n_i = n_t;
        n_t = temp;
        costhetaI = glm::abs(costhetaI);
    }
    // compute costhetaT with snells law
    float sinThetaI = glm::sqrt(glm::max(0., 1. - costhetaI * costhetaI));
    float sinThetaT = n_i / n_t * sinThetaI;
    if (sinThetaT >= 1) return 1; // total internal reflection

    float cosThetaT = glm::sqrt(1 - sinThetaT * sinThetaT);

    float r_par = ((n_t * costhetaI) - (n_i * cosThetaT)) / ((n_t * costhetaI) + (n_i * cosThetaT));
    float r_perp =  ((n_i * costhetaI) - (n_t * cosThetaT)) / ((n_i * costhetaI) + (n_t * cosThetaT));
    return (r_par * r_par + r_perp * r_perp) * 0.5;
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

    cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);

    float sin2ThetaI = 1.0f - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI * eta * eta;

    float F;

    if (sin2ThetaT >= 1.0f)
    {
        // Total internal reflection
        F = 1.0f;
    }
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

    thrust::uniform_real_distribution<float> u01(0, 1);
    float probability = u01(rng);

    // choose with biased probability
    if (probability < F) {
        resulting_color = samplePerfectSpecularReflection(
            pathSegment,
            intersect,
            normal,
            m);
        //resulting_color /= F;
    } else {
        resulting_color = samplePerfectSpecularTransmission(
            eta,
            pathSegment,
            intersect,
            normal,
            m);
        //resulting_color /= 1. - F;
    }

    return resulting_color;
}

__host__ __device__ void sampleDirectLighting(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    const Lights &l,
    thrust::default_random_engine& rng
)
{
    // get all the emitting materials
    // randomly select form the emitting material
    // sample whatever shape it is??
    return;
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{

    glm::vec3 resulting_color = m.color;

    if (m.isDielectric > 0.) { // glass, water, plastic
        resulting_color = sampleDielectric(
            pathSegment,
            intersect,
            normal,
            m,
            rng);

        pathSegment.color *= resulting_color;
    }
    // DIFFUSE
    else {
        resulting_color = sampleDiffuse(
            pathSegment,
            intersect,
            normal,
            m,
            rng); 
        pathSegment.color *= resulting_color;
    }
}
