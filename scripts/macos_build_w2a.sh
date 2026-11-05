#!/usr/bin/env bash
# Build welvet.mlx, ship dylib into welvet/native/mlx, run w2a go_v_mlx.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET="${WELVET_ROOT:-$ROOT/../welvet}"
W2A="${W2A_ROOT:-$ROOT/../w2a}"

"$ROOT/scripts/build_macos.sh"

DEST="$WELVET/native/mlx/darwin_arm64"
INC="$WELVET/native/mlx/include"
mkdir -p "$DEST" "$INC"
cp -f "$ROOT/build/libwelvet_mlx.dylib" "$DEST/"
cp -f "$ROOT/include/welvet_mlx_c.h" "$INC/"
# Also keep a copy next to the dylib for local absolute includes.
cp -f "$ROOT/include/welvet_mlx_c.h" "$DEST/"

echo "shipped → $DEST/libwelvet_mlx.dylib"

if [[ "${1:-}" == "--smoke" ]]; then
  "$ROOT/build/welvet_mlx_smoke"
  exit 0
fi

cd "$W2A"
echo "go test ./tests/go_v_mlx …"
CGO_ENABLED=1 go test ./tests/go_v_mlx -v -count=1 -timeout 60m ${@:---}
