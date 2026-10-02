#include "scene.h"

#include "utilities.h"
#include "tiny_obj_loader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".obj")
    {
        loadFromOBJ(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{

    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.isDielectric = 1.0;
            newMaterial.indexOfRefraction = 2.5; // TODO: change this later, if the json has something
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& lightsData = data["Lights"];
    for (const auto& p : lightsData)
    {
        const auto& type = p["TYPE"];
        Light newLight;

        if (p["TYPE"] == "AREA")
        {
            newLight.type = AREALIGHT;
        }
        else if (p["TYPE"] == "POINT")
        {
            newLight.type = POINTLIGHT;
            newLight.pointLight.decay = p["DECAY"];
            newLight.pointLight.range = p["RANGE"];
        }
        const auto& col = p["RGB"];
        newLight.color = glm::vec3(col[0], col[1], col[2]);
        newLight.intensity = p["INTENSITY"];

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newLight.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newLight.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newLight.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newLight.transform = utilityCore::buildTransformationMatrix(
        newLight.translation, newLight.rotation, newLight.scale);
        newLight.inverseTransform = glm::inverse(newLight.transform);
        newLight.invTranspose = glm::inverseTranspose(newLight.transform);
        newLight.normal = glm::normalize(glm::vec3(newLight.transform[1]));
        lights.push_back(newLight);


    }

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromOBJ(const std::string& filenameOBJ, const std::string& filenameMTL)
{
    // load obj wrapper referenced from https://github.com/canmom/rasteriser/blob/master/fileloader.cpp

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objmaterials;
    std::string err;
    bool success;
    if (filenameMTL.empty())
        success = tinyobj::LoadObj(&attrib, &shapes, &objmaterials, &err,
            filename.c_str(), //model to load
            nullptr, //directory to search for materials
            true); 
    else

        success = tinyobj::LoadObj(&attrib, &shapes, &objmaterials, &err,
            filename.c_str(), //model to load
            filenameMTL.c_str(), //directory to search for materials
            true); 
    
            if (!err.empty()) {
        std::cerr << err << std::endl;
    }
    if (!success) {
        exit(1);
    }

    // vertices, normal, and uvs to our format
    assert(attrib.vertices.size() == attrib.normals.size());

    // populate our own scene from the data read from tinyobj
    for (int i = 0; i < attrib.vertices.size(); i+= 3)
    {
        float v1 = attrib.vertices[i];
        float v2 = attrib.vertices[i + 1];
        float v3 = attrib.vertices[i + 2];

        float n1 = attrib.normals[i];
        float n2 = attrib.normals[i + 1];
        float n3 = attrib.normals[i + 2];
        float uv1 = attrib.texcoords[i];
        float uv2 = attrib.texcoords[i + 1];
        
        Triangle t = {
            v1, v2, v3,
            n1, n2, n3, 
            uv1, uv2
        };

        t.materialid = 1; // TODO: referring to first material
        
        geoms.push_back(t);
    }
    // TODO: handle loading materials
    Material mat = {glm::vec3(0.95), 0.f, 0,.f 0.f, 0.f, 0.f};
    materials.emplace_back(mat);
}
