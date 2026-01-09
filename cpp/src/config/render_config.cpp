/**
 * Render Configuration Implementation
 */

#include "render_config.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>

namespace vhs {
namespace config {

// Default global configuration values
RenderConfig::RenderConfig()
    : width(1920)
    , height(1080)
    , fps(30.0)
    , cq(20)
    , preset("p1")
    , use_hw_accel(true)
    , vhs_intensity(0.5f)
{
}

bool RenderConfig::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "[Config] Failed to open: %s\n", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Find key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        // Trim whitespace
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);

        // Parse known keys
        if (key == "width") width = std::stoi(value);
        else if (key == "height") height = std::stoi(value);
        else if (key == "fps") fps = std::stod(value);
        else if (key == "cq") cq = std::stoi(value);
        else if (key == "preset") preset = value;
        else if (key == "use_hw_accel") use_hw_accel = (value == "true" || value == "1");
        else if (key == "vhs_intensity") vhs_intensity = std::stof(value);
    }

    printf("[Config] Loaded: %dx%d @ %.1f fps, CQ=%d, intensity=%.2f\n",
           width, height, fps, cq, vhs_intensity);

    return true;
}

bool RenderConfig::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "[Config] Failed to write: %s\n", path.c_str());
        return false;
    }

    file << "# VHS Renderer Configuration\n\n";
    file << "# Video settings\n";
    file << "width=" << width << "\n";
    file << "height=" << height << "\n";
    file << "fps=" << fps << "\n";
    file << "\n# Encoding settings\n";
    file << "cq=" << cq << "\n";
    file << "preset=" << preset << "\n";
    file << "use_hw_accel=" << (use_hw_accel ? "true" : "false") << "\n";
    file << "\n# VHS effect settings\n";
    file << "vhs_intensity=" << vhs_intensity << "\n";

    return true;
}

} // namespace config
} // namespace vhs
