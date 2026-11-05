#pragma once

#include <cstdint>
#include <string>

#include "welvet_mlx/detect.hpp"
#include "welvet_mlx/status.hpp"

namespace welvet_mlx {

// Last MLX forward phase costs (host clocks around eval).
struct PhaseTiming {
  int64_t create_ns = 0;   // array build / cast
  int64_t finalize_ns = 0; // reserved (compile / graph cache)
  int64_t execute_ns = 0;  // mx::eval only (headline)
};

// Live MLX GPU session. Forward-only. No CPU numerical fallback.
class Session {
 public:
  Session() = default;
  ~Session() = default;

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept = default;
  Session& operator=(Session&&) noexcept = default;

  // Open GPU (Metal via MLX). Fails hard if Metal/GPU unavailable.
  static Status open_gpu(Session* out);

  bool ready() const { return ok_; }
  const std::string& device_name() const { return device_name_; }
  const std::string& mlx_version() const { return mlx_version_; }

  PhaseTiming last_phase{};

 private:
  bool ok_ = false;
  std::string device_name_;
  std::string mlx_version_;
};

} // namespace welvet_mlx
