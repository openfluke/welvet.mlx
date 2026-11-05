#!/usr/bin/env bash
# Cross-build Apple MLX (libmlx.a + mlx.metallib) for iPhoneOS arm64.
# Prerequisite: Xcode MetalToolchain
#   xcodebuild -downloadComponent MetalToolchain
#   xcrun -sdk iphoneos metal --version
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
MLX_VER="${MLX_VERSION:-0.32.4}"
MLX_SRC="${MLX_SRC:-$TP/mlx}"
METAL_CPP="${METAL_CPP:-$TP/metal_cpp}"
FMT_SRC="${FMT_SRC:-$TP/fmt}"
MLX_BUILD="${MLX_BUILD:-$TP/mlx-build-ios}"
MLX_PREFIX="${MLX_PREFIX:-$TP/mlx-ios-prefix}"
DEPLOY="${IOS_DEPLOYMENT_TARGET:-15.0}"

if ! xcrun -sdk iphoneos metal --version >/dev/null 2>&1; then
  echo "MetalToolchain missing. Install with:" >&2
  echo "  xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi

mkdir -p "$TP"

# --- sources (tarballs; avoids git-hooks issues) ---
if [[ ! -f "$MLX_SRC/CMakeLists.txt" ]]; then
  echo "fetching mlx v$MLX_VER …"
  curl -fsSL "https://github.com/ml-explore/mlx/archive/refs/tags/v${MLX_VER}.tar.gz" \
    | tar -xz -C "$TP"
  rm -rf "$MLX_SRC"
  mv "$TP/mlx-${MLX_VER}" "$MLX_SRC"
fi
if [[ ! -f "$METAL_CPP/Metal/Metal.hpp" ]]; then
  BREW_MCP="$(brew --prefix mlx 2>/dev/null)/include/metal_cpp"
  if [[ -f "$BREW_MCP/Metal/Metal.hpp" ]]; then
    mkdir -p "$METAL_CPP"
    cp -R "$BREW_MCP/." "$METAL_CPP/"
  else
    echo "Need metal_cpp headers. brew install mlx  (uses Cellar metal_cpp) or set METAL_CPP=" >&2
    exit 1
  fi
fi
if [[ ! -f "$FMT_SRC/CMakeLists.txt" ]]; then
  echo "fetching fmt 12.1.0 …"
  curl -fsSL "https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz" \
    | tar -xz -C "$TP"
  rm -rf "$FMT_SRC"
  mv "$TP/fmt-12.1.0" "$FMT_SRC"
fi

# --- patch mlx once for iOS metallib + local deps ---
python3 - "$MLX_SRC" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])

# metal_cpp local
p = src / "CMakeLists.txt"
t = p.read_text()
old = """  set(METAL_CPP_URL
      https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip)
  FetchContent_Declare(metal_cpp URL ${METAL_CPP_URL})
  FetchContent_MakeAvailable(metal_cpp)"""
new = """  set(metal_cpp_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../metal_cpp")
  if(NOT EXISTS "${metal_cpp_SOURCE_DIR}/Metal/Metal.hpp")
    message(FATAL_ERROR "metal_cpp missing at ${metal_cpp_SOURCE_DIR}")
  endif()"""
if old in t:
    t = t.replace(old, new)

old_fmt = """if(USE_SYSTEM_FMT)
  find_package(fmt REQUIRED)
else()
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(fmt)
endif()"""
new_fmt = """if(USE_SYSTEM_FMT)
  find_package(fmt REQUIRED)
else()
  set(fmt_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fmt")
  add_subdirectory("${fmt_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/fmt-build" EXCLUDE_FROM_ALL)
endif()"""
if old_fmt in t:
    t = t.replace(old_fmt, new_fmt)

needle = "  else()\n    set(MLX_METAL_VERSION 0)\n  endif()\n"
if "MLX_METAL_SDK" not in t and needle in t:
    t = t.replace(
        needle,
        needle
        + """
  if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(MLX_METAL_SDK "iphoneos" CACHE STRING "xcrun sdk for mlx metallib")
    set(MLX_METAL_MIN_FLAG "-miphoneos-version-min=")
  else()
    set(MLX_METAL_SDK "macosx" CACHE STRING "xcrun sdk for mlx metallib")
    set(MLX_METAL_MIN_FLAG "-mmacosx-version-min=")
  endif()
""",
    )
p.write_text(t)

for rel in ["cmake/extension.cmake", "mlx/backend/metal/kernels/CMakeLists.txt"]:
    p = src / rel
    t = p.read_text()
    t = t.replace("xcrun -sdk macosx metal", "xcrun -sdk ${MLX_METAL_SDK} metal")
    t = t.replace(
        "-mmacosx-version-min=${MTLLIB_DEPLOYMENT_TARGET}",
        "${MLX_METAL_MIN_FLAG}${MTLLIB_DEPLOYMENT_TARGET}",
    )
    t = t.replace(
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}",
        "${MLX_METAL_MIN_FLAG}${CMAKE_OSX_DEPLOYMENT_TARGET}",
    )
    p.write_text(t)
print("mlx iOS patches applied")
PY

SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CXX="$(xcrun --sdk iphoneos --find clang++)"
rm -rf "$MLX_BUILD" "$MLX_PREFIX"
cmake -S "$MLX_SRC" -B "$MLX_BUILD" -G "Unix Makefiles" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY" \
  -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$MLX_PREFIX" \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_CPU=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_METAL_SDK=iphoneos \
  -DFMT_INSTALL=OFF

cmake --build "$MLX_BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --install "$MLX_BUILD"

# Ensure metallib lands next to lib for runtime lookup.
find "$MLX_BUILD" -name 'mlx.metallib' -exec cp -f {} "$MLX_PREFIX/lib/" \; 2>/dev/null || true

echo "MLX iOS prefix: $MLX_PREFIX"
ls -la "$MLX_PREFIX/lib" "$MLX_PREFIX/include/mlx" | head
