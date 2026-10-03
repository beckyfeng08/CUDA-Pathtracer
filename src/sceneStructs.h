#pragma once

#include <cuda_runtime.h>
#include "utilities.h"

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE
};


enum LightType
{
    AREALIGHT,
    POINTLIGHT
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};



struct Geom
{
    enum GeomType type;
    int materialid;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    __host__ __device__
    Geom(GeomType type)
        : type(type),
          materialid(0),
          translation(0.f),
          rotation(0.f),
          scale(1.f)
    {
        transform = utilityCore::buildTransformationMatrix(
              translation, rotation, scale);
        inverseTransform = glm::inverse(transform);
        invTranspose = glm::transpose(glm::inverse(transform));
    }
};

struct Triangle : Geom
{
    glm::vec3 v1, v2, v3;
    glm::vec3 normal;
    bool hasUVs;
    glm::vec2 uv1, uv2, uv3;
    glm::vec3 centroid;

    struct {
        glm::vec3 minCorner;
        glm::vec3 maxCorner;
    } bbox;

    Triangle() : Geom(TRIANGLE)
    {}

    Triangle(
        glm::vec3 _v1,
        glm::vec3 _v2,
        glm::vec3 _v3,
        glm::vec2 _uv1,
        glm::vec2 _uv2,
        glm::vec2 _uv3)
        : Geom(TRIANGLE),
          v1(_v1),
          v2(_v2),
          v3(_v3),
          uv1(_uv1),
          uv2(_uv2),
          uv3(_uv3),
          hasUVs(true)
    {
        centroid = (v1 + v2 + v3) / 3.f;

        bbox.minCorner = glm::min(v1, glm::min(v2, v3));
        bbox.maxCorner = glm::max(v1, glm::max(v2, v3));

        normal = glm::cross(v2 - v1, v3 - v1);
    }

    Triangle(
        glm::vec3 _v1,
        glm::vec3 _v2,
        glm::vec3 _v3)
        : Geom(TRIANGLE),
        hasUVs(false),
          v1(_v1),
          v2(_v2),
          v3(_v3)
    {
        centroid = (v1 + v2 + v3) / 3.f;

        bbox.minCorner = glm::min(v1, glm::min(v2, v3));
        bbox.maxCorner = glm::max(v1, glm::max(v2, v3));

        normal = glm::cross(v2 - v1, v3 - v1);
    }
};

struct Material
{
    glm::vec3 color;
    float hasReflective;
    float hasRefractive;
    float isDielectric;

    float indexOfRefraction;
    float emittance;
};

struct Light
{
    glm::vec3 color;


    struct {
        float decay;
        float range;
    } pointLight;

    enum LightType type;

    float intensity;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    glm::vec3 normal;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 radiance;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int isLight;
  int lightId;
};
