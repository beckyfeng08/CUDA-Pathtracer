#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void set_up_camera_default(int resx,
        int resy,
        float fovy,
        int iterations,
        int depth,
        const std::string& fileName,
        glm::vec3 camPos,
        glm::vec3 camLookAt,
        glm::vec3 camUp
    );
    void set_up_default_lights(glm::vec3 color);
    void set_up_render_cam(Camera& camera, RenderState& state);
    void loadFromOBJ(const std::string& filenameOBJ, const std::string& filenameMTL);

public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;
    RenderState state;
};
