#!/usr/bin/env bash
# Build welvet.mlx for iPhoneOS arm64 and install into Go Welvet native/mlx/ios_arm64.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi

# Build Apple MLX for iphoneos if prefix missing.
PREFIX="$ROOT/third_party/mlx-ios-prefix"
if [[ -z "${MLX_ROOT:-}" ]]; then
  if [[ ! -f "$PREFIX/lib/libmlx.a" ]]; then
    "$ROOT/scripts/build_mlx_ios.sh"
  fi
  export MLX_ROOT="$PREFIX"
fi

"$ROOT/scripts/build_ios.sh"

DEST="$WELVET_GO_ROOT/native/mlx/ios_arm64"
INC="$WELVET_GO_ROOT/native/mlx/include"
mkdir -p "$DEST" "$INC"
cp -f "$ROOT/build-ios-arm64/libwelvet_mlx.a" "$DEST/libwelvet_mlx.a"
cp -f "$ROOT/include/welvet_mlx_c.h" "$INC/welvet_mlx_c.h"

# Ship MLX dependency artifacts required to link/run on device.
if [[ -f "$MLX_ROOT/lib/libmlx.a" ]]; then
  cp -f "$MLX_ROOT/lib/libmlx.a" "$DEST/libmlx.a"
fi
if [[ -f "$MLX_ROOT/lib/mlx.metallib" ]]; then
  cp -f "$MLX_ROOT/lib/mlx.metallib" "$DEST/mlx.metallib"
fi

VERSION="$(sed -n 's/.*project(welvet_mlx VERSION \([0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt" | head -1)"
VERSION="${VERSION:-1.0.0}"

{
  echo "engine=welvet.mlx"
  echo "version=$VERSION"
  echo "artifact=libwelvet_mlx.a"
  echo "platform=ios_arm64"
  echo "minimum_ios=${IOS_DEPLOYMENT_TARGET:-15.0}"
  echo "mlx_root=$MLX_ROOT"
  echo "source_checkout=$ROOT"
  shasum -a 256 "$DEST/libwelvet_mlx.a"
  [[ -f "$DEST/libmlx.a" ]] && shasum -a 256 "$DEST/libmlx.a"
  [[ -f "$DEST/mlx.metallib" ]] && shasum -a 256 "$DEST/mlx.metallib"
} > "$DEST/MANIFEST.txt"

if [[ -f "$WELVET_GO_ROOT/native/mlx/MANIFEST.txt" ]]; then
  if ! grep -q 'ios_arm64=present' "$WELVET_GO_ROOT/native/mlx/MANIFEST.txt"; then
    echo "ios_arm64=present" >> "$WELVET_GO_ROOT/native/mlx/MANIFEST.txt"
  fi
fi

cat > "$DEST/README.md" <<EOF
# ios_arm64

Shipped static \`libwelvet_mlx.a\` (+ \`libmlx.a\` / \`mlx.metallib\`) for iPhoneOS arm64
(deployment ${IOS_DEPLOYMENT_TARGET:-15.0}+). Link with \`-force_load\` plus
\`-framework Metal -framework Foundation -framework Accelerate\`; bundle \`mlx.metallib\`.

\`\`\`bash
cd welvet.mlx
# once: xcodebuild -downloadComponent MetalToolchain
./scripts/ios_install_go.sh
\`\`\`

Go \`nativemlx\` cgo path is macOS dylib today; this artifact is for Xcode / gomobile app link.
EOF

echo "Installed $DEST/libwelvet_mlx.a"
