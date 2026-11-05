#pragma once

#include <string>
#include <vector>

namespace welvet_mlx {

struct DetectResult {
  bool mlx_found = false;
  bool metal_available = false;
  bool gpu_available = false;
  std::string mlx_version;
  std::string device_name;
  std::string note;
  std::vector<std::string> hints;
};

DetectResult detect();

} // namespace welvet_mlx
