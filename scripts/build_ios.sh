#!/usr/bin/env bash
# iPhoneOS arm64 static build for welvet.mlx (libwelvet_mlx.a).
# Requires an iphoneos-built Apple MLX prefix (libmlx.a + headers + metallib).
#
#   ./scripts/build_mlx_ios.sh          # builds third_party MLX → mlx-ios-prefix
#   MLX_ROOT=... ./scripts/build_ios.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-ios-arm64}"
DEPLOY="${IOS_DEPLOYMENT_TARGET:-15.0}"
DEFAULT_PREFIX="$ROOT/third_party/mlx-ios-prefix"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "welvet.mlx build_ios.sh requires macOS + Xcode" >&2
  exit 1
fi

if [[ -z "${MLX_ROOT:-}" ]]; then
  if [[ -f "$DEFAULT_PREFIX/include/mlx/mlx.h" && -f "$DEFAULT_PREFIX/lib/libmlx.a" ]]; then
    export MLX_ROOT="$DEFAULT_PREFIX"
  else
    echo "MLX_ROOT not set and $DEFAULT_PREFIX missing." >&2
    echo "Run: ./scripts/build_mlx_ios.sh   (needs MetalToolchain: xcodebuild -downloadComponent MetalToolchain)" >&2
    exit 1
  fi
fi
if [[ ! -f "$MLX_ROOT/include/mlx/mlx.h" ]]; then
  echo "MLX_ROOT=$MLX_ROOT has no include/mlx/mlx.h" >&2
  exit 1
fi
if [[ ! -f "$MLX_ROOT/lib/libmlx.a" && ! -f "$MLX_ROOT/lib/libmlx.dylib" ]]; then
  echo "MLX_ROOT=$MLX_ROOT has no lib/libmlx.a (iphoneos static preferred)" >&2
  exit 1
fi

SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CXX="$(xcrun --sdk iphoneos --find clang++)"

cmake -S "$ROOT" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY" \
  -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="-isysroot $SDK -arch arm64 -miphoneos-version-min=$DEPLOY" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_ROOT="$MLX_ROOT" \
  -DWELVET_MLX_BUILD_SHARED=OFF \
  -DWELVET_MLX_BUILD_TOOLS=OFF \
  "$@"

cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "built: $BUILD/libwelvet_mlx.a"
lipo -info "$BUILD/libwelvet_mlx.a" 2>/dev/null || true
