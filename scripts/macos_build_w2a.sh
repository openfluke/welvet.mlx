#!/usr/bin/env bash
# Build welvet.mlx, ship dylib into welvet/native/mlx, run w2a go_v_mlx.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET="${WELVET_ROOT:-$ROOT/../welvet}"
W2A="${W2A_ROOT:-$ROOT/../w2a}"

# Build + install into sibling Go Welvet (dylib, header, MANIFEST).
"$ROOT/scripts/macos_install_go.sh"

echo "shipped → $WELVET/native/mlx/darwin_arm64/libwelvet_mlx.dylib"

if [[ "${1:-}" == "--smoke" ]]; then
  "$ROOT/build/welvet_mlx_smoke"
  exit 0
fi

cd "$W2A"
echo "go test ./tests/go_v_mlx …"
CGO_ENABLED=1 go test ./tests/go_v_mlx -v -count=1 -timeout 60m ${@:---}
