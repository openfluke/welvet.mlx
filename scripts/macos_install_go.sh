#!/usr/bin/env bash
# Build and install welvet.mlx into a local Go Welvet checkout (native/mlx).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi

"$ROOT/scripts/build_macos.sh" -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0

DEST="$WELVET_GO_ROOT/native/mlx/darwin_arm64"
INC="$WELVET_GO_ROOT/native/mlx/include"
mkdir -p "$DEST" "$INC"
cp -f "$ROOT/build/libwelvet_mlx.dylib" "$DEST/libwelvet_mlx.dylib"
cp -f "$ROOT/include/welvet_mlx_c.h" "$INC/welvet_mlx_c.h"
# Keep a copy beside the dylib for absolute local includes.
cp -f "$ROOT/include/welvet_mlx_c.h" "$DEST/welvet_mlx_c.h"

VERSION="$(sed -n 's/.*project(welvet_mlx VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt" | head -1)"
if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/.*return "\([0-9][0-9.]*\)".*/\1/p' "$ROOT/src/c_abi.cpp" | head -1)"
fi
VERSION="${VERSION:-unknown}"

{
  echo "engine=welvet.mlx"
  echo "version=$VERSION"
  echo "model_abi=1"
  echo "platform=darwin_arm64"
  echo "minimum_macos=13.0"
  echo "source_checkout=$ROOT"
  shasum -a 256 "$DEST/libwelvet_mlx.dylib"
} > "$WELVET_GO_ROOT/native/mlx/MANIFEST.txt"

echo "Installed $DEST/libwelvet_mlx.dylib"
echo "Header    $INC/welvet_mlx_c.h"
echo "Go build: CGO_ENABLED=1 go build ./nativemlx"
