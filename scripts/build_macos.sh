#!/usr/bin/env bash
# macOS Apple silicon host build for welvet.mlx (MLX C++ / Metal GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "welvet.mlx build_macos.sh requires macOS" >&2
  exit 1
fi

if [[ -z "${MLX_ROOT:-}" ]]; then
  for cand in /opt/homebrew /usr/local; do
    if [[ -f "$cand/include/mlx/mlx.h" && -f "$cand/lib/libmlx.dylib" ]]; then
      export MLX_ROOT="$cand"
      break
    fi
  done
fi
if [[ -z "${MLX_ROOT:-}" ]]; then
  echo "MLX not found. brew install mlx  (or set MLX_ROOT). No stub build." >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMLX_ROOT="$MLX_ROOT" "$@"
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "built: $BUILD"
echo "  lib:  $BUILD/libwelvet_mlx.dylib"
echo "  smoke: $BUILD/welvet_mlx_smoke"
echo
echo "smoke:  $BUILD/welvet_mlx_smoke"
echo "ship:   cp $BUILD/libwelvet_mlx.dylib ../welvet/native/mlx/darwin_arm64/"
echo "        cp include/welvet_mlx_c.h ../welvet/native/mlx/include/"
