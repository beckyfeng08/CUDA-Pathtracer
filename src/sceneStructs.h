#pragma once

#include <cuda_runtime.h>

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
    Geom(GeomType type) : type(type) {}
};

struct Triangle : Geom
{
    // the ints are indices to scene.vertices, scene.normals, and scene.uvs
     Triangle() : Geom(TRIANGLE) {}
     Triangle(int v1,
         int v2,
         int v3,
         int n1,
         int n2,
         int n3,
         int uv1,
         int uv2,
         int uv3
         ) : Geom(TRIANGLE),
         v1(v1), v2(v2), v3(v3), 
         n1(n1), n2(n2), n3(n3), 
         uv1(uv1), uv2(uv2), uv3(uv3)
     {}

    // vertices
    int v1, v2, v3,
    // normals
    n1, n2, n3,
    // uvs if need be
    uv1, uv2, uv3;

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
