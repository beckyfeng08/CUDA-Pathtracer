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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float probability = u01(rng);

    // depending on material properties, choose between m.reflective, refractive, and diffuse
    int num_properties = (int)m.hasReflective + (int)m.hasRefractive + 1;
    float reflect_prob = (float)m.hasReflective / (float)num_properties;
    float refract_prob = (float)m.hasRefractive / (float)num_properties;

    glm::vec3 resulting_color = m.color;

    // REFLECTIVE
    if (probability < reflect_prob) {
        // get the new direction by reflecting the ray direction by the normal
        // sample around a lobe described by specular, not completely reflective in this case???

        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + normal * EPSILON;
        resulting_color /= reflect_prob;
    } 
    // REFRACTIVE
    else if (probability < refract_prob + reflect_prob) {
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

        // returns 0 for total internal reflection, so reflection occurs here
        if (glm::length(refraction_dir) < EPSILON) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, n);
            pathSegment.ray.origin = intersect + n * EPSILON; // make sure it doesn't self intersect, stay within current medium
        } else {
            // refraction occurs here
            pathSegment.ray.direction = glm::normalize(refraction_dir);
            pathSegment.ray.origin = intersect - n * EPSILON; // make sure it doesn't self intersect, stay within outgoing medium
        }
        resulting_color /= refract_prob;

    } 
    // DIFFUSE
    else {
        // diffuse
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON; // add some offset so it doesn't self intersect
        resulting_color /= (1. - (refract_prob + reflect_prob));
    }

    pathSegment.color *= resulting_color;

}
