#include "scene.h"

#include "sceneStructs.h"
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


void Scene::set_up_camera_default(int resx, 
                                int resy, 
                                float fovy, 
                                int iterations,
                                int depth,
                                std::string fileName,
                                glm::vec3 camPos,
                                glm::vec3 camLookAt,
                                glm::vec3 camUp
)
{
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = resx;
    camera.resolution.y = resy;
    float fovy = fovy;
    state.iterations = iterations;
    state.traceDepth = depth;
    state.imageName = fileName;

    camera.position = camPos;
    camera.lookAt = camLookAt;
    camera.up = camUp;

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

}

void Scene::set_up_default_lights(glm::vec3 color)
{
    // modify as necessary. There is a point and area light in here
    Light default_area_light;
    default_area_light.type = AREALIGHT;
    default_area_light.color = color;
    default_area_light.intensity = 10;

    default_area_light.translation = glm::vec3(0.0, 9.99, 0.0);
    default_area_light.rotation = glm::vec3(180.0, 0.0, 0.0);
    default_area_light.scale = glm::vec3(1.f);

    default_area_light.transform = utilityCore::buildTransformationMatrix(
        default_area_light.translation, default_area_light.rotation, default_area_light.scale);
    default_area_light.inverseTransform = glm::inverse(default_area_light.transform);
    default_area_light.invTranspose = glm::inverseTranspose(default_area_light.transform);
    default_area_light.normal = glm::normalize(glm::vec3(default_area_light.transform[1]));


    Light default_point_light;
    default_point_light.type = POINTLIGHT;
    default_point_light.pointLight.decay = 10.0;
    default_point_light.pointLight.range = 20.0;
    default_point_light.color = color;
    default_point_light.intensity = 10;
    default_point_light.translation = glm::vec3(-3.0, 4.99, 0.0);
    default_point_light.rotation = glm::vec3(0.0, 0.0, 0.0);
    default_point_light.scale = glm::vec3(1.f);

    default_point_light.transform = utilityCore::buildTransformationMatrix(
        default_point_light.translation, default_point_light.rotation, default_point_light.scale);
    default_point_light.inverseTransform = glm::inverse(default_point_light.transform);
    default_point_light.invTranspose = glm::inverseTranspose(default_point_light.transform);
    default_point_light.normal = glm::normalize(glm::vec3(default_point_light.transform[1]));

    light.push_back(default_area_light);
    lights.push_back(default_point_light);
}

void Scene::set_up_render_cam(Camera& camera, RenderState& state)
{
    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

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

    set_up_render_cam(camera, state);

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

    // TODO: handle loading materials
    // tbh idc im just gonna make it diffuse
    Material mat = {glm::vec3(0.95), 0.f, 0.f, 0.f, 0.f, 0.f};
    materials.emplace_back(mat);

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

        t.materialid = 0; // TODO: referring to first material, change if we do better material support

        // triangle does not need these attributes but lets populate them in case something happens
        t.translation = glm::vec3(0.f);
        t.rotation = glm::vec3(0.f);
        t.scale = glm::vec3(1.f);
        t.transform = utilityCore::buildTransformationMatrix(
            t.translation, t.rotation, t.scale);
        t.inverseTransform = glm::inverse(t.transform);
        t.invTranspose = glm::inverseTranspose(t.transform);

        geoms.push_back(t);
    }
   
    // hardcoded lights and camera in scene

    set_up_default_lights();

    set_up_camera_default(800, 800, 45.f, 5000, 8, filenameOBJ, 
                                glm::vec3(0.f, 5.f, 10.5),
                                glm::vec3(0.f, 5.f, 0.f),
                                glm::vec3(0.f, 1.f, 0.f));

    //required for display: set up render camera stuff
    set_up_render_cam(camera, state);
}
