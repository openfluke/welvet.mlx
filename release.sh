#!/usr/bin/env bash
# Build, validate, package and publish this independent Apple MLX backend.
# Default: commit local source changes, push the current branch + version tag,
# then create a GitHub Release with macOS arm64 shared/static binaries.
# The script never force-pushes or moves an existing tag.
#
# iOS artifacts are NOT packaged here (mlx.metallib ~140MB) — see
# scripts/ios_install_go.sh / sibling welvet native/mlx/ios_arm64/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DRY_RUN=0
PLAN=0
SKIP_BUILD=0
SKIP_TESTS=0
TEST_GO=0
DRAFT=0
AUTO_COMMIT=1
BUILD_ROOT="$ROOT/build-release"
DIST="$ROOT/dist"
MIN_MACOS="13.0"

usage() {
  cat <<'HELP'
Usage: ./release.sh [options]
  --dry-run       Build, test and package; do not commit, tag, push or publish
  --plan          Show version and release actions only; no builds or mutations
  --skip-build    Reuse build-release/shared and build-release/static
  --skip-tests    Skip native GPU smoke (recorded in release notes)
  --test-go       Also run sibling w2a go_v_mlx suite
  --no-commit     Require a clean source tree instead of committing local changes
  --draft         Create a draft GitHub Release after pushing branch/tag
  -h, --help      Show this help

Default: release v<current CMake/README/runtime version> to the origin GitHub repo.
Requires macOS Apple silicon, CMake, Apple Clang/SDK, Homebrew mlx (or MLX_ROOT),
Python 3 and git. Publishing additionally requires an authenticated GitHub CLI (gh).
Generated builds and dist/ are ignored. No sibling repository is modified.
HELP
}
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --plan) PLAN=1; DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --test-go) TEST_GO=1 ;;
    --no-commit) AUTO_COMMIT=0 ;;
    --draft) DRAFT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command missing: $1"; }
need git
need python3

VERSION="$(python3 - "$ROOT" <<'PY'
from pathlib import Path
import re, sys
r = Path(sys.argv[1])
cm = re.search(r'project\(welvet_mlx\s+VERSION\s+([0-9.]+)', (r / 'CMakeLists.txt').read_text())
rm = re.search(r'\*\*Library\*\*\s*\|\s*\*\*([0-9.]+)\*\*', (r / 'README.md').read_text())
abi = re.search(r'wmlx_version\(void\)\s*\{\s*return\s+"([0-9.]+)"', (r / 'src/c_abi.cpp').read_text())
if not all((cm, rm, abi)) or len({m.group(1) for m in (cm, rm, abi)}) != 1:
    raise SystemExit('CMake, README (**Library**) and wmlx_version() must agree')
print(cm.group(1))
PY
)"

TAG="v$VERSION"
BRANCH="$(git symbolic-ref --quiet --short HEAD)" || fail "Check out a branch before releasing"
REPO="$(git remote get-url origin | python3 -c '
import re, sys
url = sys.stdin.read().strip()
m = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?", url)
if not m:
    raise SystemExit("origin must point to a GitHub repository")
print(m.group(1))
')"
[[ "$REPO" == */welvet.mlx ]] || fail "origin must name a welvet.mlx repository"

if [[ "$DRY_RUN" -eq 0 ]]; then
  need gh
  gh auth status --hostname github.com >/dev/null 2>&1 || fail "Run gh auth login first"
  if [[ "$AUTO_COMMIT" -eq 0 && -n "$(git status --porcelain)" ]]; then
    fail "Source tree is dirty; commit changes or omit --no-commit"
  fi
  if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    [[ "$(git rev-parse "$TAG^{commit}")" == "$(git rev-parse HEAD)" ]] || fail "$TAG already names another commit; tags are never moved"
    [[ -z "$(git status --porcelain)" ]] || fail "$TAG already exists; do not change the tagged source"
  fi
fi

printf 'welvet.mlx %s\nGitHub: %s\nBranch: %s\nAsset: dist/welvet-mlx-%s-macos-arm64.tar.gz\n' \
  "$TAG" "$REPO" "$BRANCH" "$TAG"

if [[ "$PLAN" -eq 1 ]]; then
  echo "Plan: build shared + static → install → GPU smoke (forward-only) → package + SHA256"
  [[ "$TEST_GO" -eq 0 ]] || echo "Plan: test sibling w2a go_v_mlx"
  echo "Publish: commit local source (unless --no-commit), push branch + immutable tag, gh release create"
  echo "Note: iOS artifacts are not packaged (mlx.metallib too large)"
  echo "Plan only: no files, git state or remote state changed"
  exit 0
fi

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail "Build/package requires Apple silicon macOS"
for command in cmake xcrun clang++ tar shasum otool; do need "$command"; done

# Resolve MLX (Homebrew or MLX_ROOT)
if [[ -z "${MLX_ROOT:-}" ]]; then
  for cand in /opt/homebrew /usr/local; do
    if [[ -f "$cand/include/mlx/mlx.h" && -f "$cand/lib/libmlx.dylib" ]]; then
      export MLX_ROOT="$cand"
      break
    fi
  done
fi
[[ -n "${MLX_ROOT:-}" && -f "$MLX_ROOT/include/mlx/mlx.h" ]] || fail "MLX not found; brew install mlx or set MLX_ROOT"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
mkdir -p "$BUILD_ROOT" "$DIST"
STAGE="$(mktemp -d "$DIST/.stage.XXXXXX")"
cleanup() {
  python3 - "$STAGE" <<'PY'
from pathlib import Path
import shutil, sys
p = Path(sys.argv[1])
if p.name.startswith('.stage.'):
    shutil.rmtree(p, ignore_errors=True)
PY
}
trap cleanup EXIT
NAME="welvet-mlx-$TAG-macos-arm64"
PACKAGE="$STAGE/$NAME"
mkdir -p "$PACKAGE"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  cmake -S "$ROOT" -B "$BUILD_ROOT/shared" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS" \
    -DMLX_ROOT="$MLX_ROOT" \
    -DWELVET_MLX_BUILD_SHARED=ON \
    -DWELVET_MLX_BUILD_TOOLS=ON
  cmake --build "$BUILD_ROOT/shared" --parallel "$JOBS"

  cmake -S "$ROOT" -B "$BUILD_ROOT/static" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS" \
    -DMLX_ROOT="$MLX_ROOT" \
    -DWELVET_MLX_BUILD_SHARED=OFF \
    -DWELVET_MLX_BUILD_TOOLS=OFF
  cmake --build "$BUILD_ROOT/static" --parallel "$JOBS"
fi

for artifact in shared/libwelvet_mlx.dylib shared/welvet_mlx_smoke static/libwelvet_mlx.a; do
  [[ -f "$BUILD_ROOT/$artifact" ]] || fail "Missing $BUILD_ROOT/$artifact; run without --skip-build"
done

cmake --install "$BUILD_ROOT/shared" --prefix "$PACKAGE"
cmake --install "$BUILD_ROOT/static" --prefix "$PACKAGE"

# Prefer installed names; CMake may put shared as lib/libwelvet_mlx.dylib and static as .a
[[ -f "$PACKAGE/lib/libwelvet_mlx.dylib" || -f "$PACKAGE/lib/libwelvet_mlx.so" ]] \
  || [[ -f "$BUILD_ROOT/shared/libwelvet_mlx.dylib" ]] || fail "Shared lib not installed"
# Ensure both shared + static land in package lib/
cp -f "$BUILD_ROOT/shared/libwelvet_mlx.dylib" "$PACKAGE/lib/libwelvet_mlx.dylib"
cp -f "$BUILD_ROOT/static/libwelvet_mlx.a" "$PACKAGE/lib/libwelvet_mlx.a"
mkdir -p "$PACKAGE/bin"
cp -f "$BUILD_ROOT/shared/welvet_mlx_smoke" "$PACKAGE/bin/welvet_mlx_smoke"
chmod +x "$PACKAGE/bin/welvet_mlx_smoke"

SMOKE_VER="$("$PACKAGE/bin/welvet_mlx_smoke" --version 2>/dev/null | head -1 || true)"
[[ "$SMOKE_VER" == *"$VERSION"* ]] || fail "Packaged smoke version mismatch (got: ${SMOKE_VER:-empty})"

NATIVE_CHECK="skipped (--skip-tests)"
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  # Full smoke needs Metal GPU; session-open failure is a hard Unavailable (no CPU fallback).
  set +e
  "$PACKAGE/bin/welvet_mlx_smoke" >/tmp/welvet_mlx_release_smoke.log 2>&1
  smoke_rc=$?
  set -e
  if [[ "$smoke_rc" -eq 0 ]]; then
    NATIVE_CHECK="passed: packaged C ABI + MLX GPU forward smoke"
  elif [[ "$smoke_rc" -eq 2 ]] || grep -q 'GPU session open failed' /tmp/welvet_mlx_release_smoke.log; then
    fail "GPU smoke: MLX GPU unavailable (no CPU fallback). Fix Metal/MLX or pass --skip-tests. Log: /tmp/welvet_mlx_release_smoke.log"
  else
    echo "---- smoke log (tail) ----" >&2
    tail -40 /tmp/welvet_mlx_release_smoke.log >&2 || true
    fail "GPU smoke failed (exit $smoke_rc). Log: /tmp/welvet_mlx_release_smoke.log"
  fi
fi

GO_CHECK="not requested"
if [[ "$TEST_GO" -eq 1 ]]; then
  need go
  W2A="${WELVET_W2A_ROOT:-$ROOT/../w2a}"
  WELVET_GO="${WELVET_GO_ROOT:-$ROOT/../welvet}"
  [[ -f "$W2A/go.mod" ]] || fail "Sibling w2a missing; set WELVET_W2A_ROOT or omit --test-go"
  [[ -f "$WELVET_GO/go.mod" ]] || fail "Sibling welvet missing; set WELVET_GO_ROOT or omit --test-go"
  # Ship dylib into Go native tree for the suite, then test.
  DEST="$WELVET_GO/native/mlx/darwin_arm64"
  mkdir -p "$DEST" "$WELVET_GO/native/mlx/include"
  cp -f "$PACKAGE/lib/libwelvet_mlx.dylib" "$DEST/"
  cp -f "$ROOT/include/welvet_mlx_c.h" "$WELVET_GO/native/mlx/include/"
  (
    cd "$W2A"
    CGO_ENABLED=1 go test ./tests/go_v_mlx -count=1 -timeout 60m -v
  )
  GO_CHECK="passed: w2a go_v_mlx"
fi

if [[ "$DRY_RUN" -eq 0 && -n "$(git status --porcelain)" ]]; then
  [[ "$AUTO_COMMIT" -eq 1 ]] || fail "Dirty source tree"
  git add -A
  git commit -m "Release welvet.mlx $TAG"
fi
COMMIT="$(git rev-parse HEAD)"
DIRTY=false
[[ -z "$(git status --porcelain)" ]] || DIRTY=true

cp "$ROOT/README.md" "$ROOT/VERSIONING.md" "$ROOT/LICENSE" "$ROOT/GAPS.md" "$PACKAGE/"
mkdir -p "$PACKAGE/docs"
cp "$ROOT/docs/MLX_ONLY.md" "$ROOT/docs/FORWARD_CHECKLIST.md" "$PACKAGE/docs/"

python3 - "$PACKAGE" "$VERSION" "$COMMIT" "$DIRTY" "$MIN_MACOS" "$NATIVE_CHECK" "$GO_CHECK" "$MLX_ROOT" <<'PY'
import json, subprocess, sys
from pathlib import Path
p, v, commit, dirty, minimum, native, go, mlx_root = sys.argv[1:]
manifest = dict(
    engine='welvet.mlx',
    version=v,
    platform='macos-arm64',
    minimum_macos=minimum,
    model_abi=1,
    forward_only=True,
    ios_artifacts='not_packaged_too_large',
    mlx_root_at_build=mlx_root,
    runtime_depends=['libmlx.dylib (Homebrew / MLX_ROOT)'],
    source_commit=commit,
    source_dirty_at_build=dirty == 'true',
    native_validation=native,
    go_validation=go,
    sdk_version=subprocess.check_output(['xcrun', '--show-sdk-version'], text=True).strip(),
    compiler=subprocess.check_output(['clang++', '--version'], text=True).splitlines()[0],
)
Path(p, 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY

ARCHIVE="$DIST/$NAME.tar.gz"
COPYFILE_DISABLE=1 tar -czf "$ARCHIVE" -C "$STAGE" "$NAME"
(cd "$DIST" && shasum -a 256 "$NAME.tar.gz" > "$NAME.sha256")
NOTES="$DIST/release-$TAG.md"
cat > "$NOTES" <<NOTES
# welvet.mlx $TAG

Independent **forward-only** Apple MLX GPU backend for Welvet (libmlx / Metal under the hood).
No CPU fallback. No train / backward. See VERSIONING.md and docs/FORWARD_CHECKLIST.md.

- 14 signed forward leaves × caps dtypes (f32/f16/bf16 + ints/c64 where capped)
- C ABI \`welvet_mlx_c.h\` for Welvet Go \`nativemlx\` / \`BackendMLX\`
- Dense prepare/exec; Sequential = host ping-pong (unfused)
- Runtime dependency: **libmlx** (Homebrew \`mlx\` or \`MLX_ROOT\`) — not bundled

## macOS arm64 artifact

Deployment target: macOS $MIN_MACOS. Includes shared + static \`libwelvet_mlx\`, headers,
smoke CLI, license, docs and provenance manifest.

**iOS** artifacts are **not** attached (Apple MLX \`mlx.metallib\` ~140MB). Build locally with
\`scripts/ios_install_go.sh\` if needed.

Validation: $NATIVE_CHECK. Go/w2a: $GO_CHECK.
Source commit: $COMMIT (dirty working-tree package: $DIRTY).

Verify: \`shasum -a 256 -c $NAME.sha256\`
NOTES

printf '\nPrepared:\n  %s\n  %s\n  %s\n' "$ARCHIVE" "$DIST/$NAME.sha256" "$NOTES"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete: no commit, tag, push, authentication check or GitHub release performed"
  exit 0
fi

if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  [[ "$(git rev-parse "$TAG^{commit}")" == "$COMMIT" ]] || fail "$TAG is not the release commit"
else
  git tag -a "$TAG" -m "welvet.mlx $TAG"
fi
git push origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"
release_args=("$TAG" "$ARCHIVE" "$DIST/$NAME.sha256" --repo "$REPO" --verify-tag
  --title "welvet.mlx $TAG" --notes-file "$NOTES")
[[ "$DRAFT" -eq 0 ]] || release_args+=(--draft)
gh release create "${release_args[@]}"
echo "Release: https://github.com/$REPO/releases/tag/$TAG"
