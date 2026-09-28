#include "interactions.h"

#include "utilities.h"
#include <glm/gtc/constants.hpp>
#include <thrust/random.h>

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

// TODO
__host__ __device__ glm::vec3 samplePerfectSpecularTransmission(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m) 
{
    
 // from air to medium
    float n_incident = 1.0;
    float n_outgoing =  m.indexOfRefraction;
    glm::vec3 n = normal;

    //check to see if we are entering or exiting the (supposedly thick) material
    float incident_dot_normal = glm::dot(normal, pathSegment.ray.direction);
    if (incident_dot_normal > 0.0) { // from medium to air
        n_incident = n_outgoing;
        n_outgoing = 1.0;
        n = -n; // normal is negative if we are in the medium (since normal points out to air)
    }

    // check to see if we are refracting or reflecting from this angle
    glm::vec3 refraction_dir = glm::refract(glm::normalize(pathSegment.ray.direction), glm::normalize(n), n_incident/n_outgoing);

    pathSegment.ray.direction = glm::normalize(refraction_dir);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON; // make sure it doesn't self intersect, stay within outgoing medium

    return m.color;
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
    thrust::default_random_engine &rng
)
{
    glm::vec3 resulting_color = m.color;
    
    float n_i = 1.0;
    float n_t = m.indexOfRefraction;
    glm::vec3 n = normal;

    float costhetaI = glm::dot(n, pathSegment.ray.direction);
    
    // determine if ray exits the dielectric medium or enters
    if (costhetaI > 0.f) {
        float temp = n_i;
        n_i = n_t;
        n_t = temp;
        costhetaI = glm::abs(costhetaI);
        n = -normal;
    }

    float F = computeFresnelReflectance(costhetaI, n_i, n_t);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float probability = u01(rng);

    // TODO: uncomment when done
    // choose with biased probability
    //if (probability < F) {
       /* resulting_color = samplePerfectSpecularReflection(
            pathSegment,
            intersect,
            normal,
            m);
        resulting_color /= F;*/
    //} else {
        resulting_color = samplePerfectSpecularTransmission(
            pathSegment,
            intersect,
            normal,
            m);
       
        //resulting_color /= 1. - F; // MAKE SURE DENOM IS NOT 0
    //}
    return resulting_color;
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
