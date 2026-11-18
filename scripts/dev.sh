#!/usr/bin/env bash
# Project-local entry point; never changes the parent shell or installs packages.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VISION_PREFIX=/opt/miniforge3/envs/rk3588-infer
ROLE=vision
ROLE_SET=0

usage() {
    cat <<'EOF'
Usage: bash scripts/dev.sh [--role system|vision] shell
       bash scripts/dev.sh [--role system|vision] python [Python arguments...]
       bash scripts/dev.sh doctor
       bash scripts/dev.sh build
       bash scripts/dev.sh test

Default Python/shell role: vision (existing rk3588-infer environment).
system uses /usr/bin/python3. doctor always checks both roles.
build uses the existing native production preset; test builds the test preset
and runs the CPU-only regressions selected by the cpu CTest label.
No command installs or upgrades packages.
Exit the project shell with exit. --role is valid only for shell/python.
EOF
}

fail() { printf 'yolo dev: %s\n' "$*" >&2; exit 2; }

while [[ ${1:-} == --role ]]; do
    [[ $# -ge 2 && $ROLE_SET == 0 ]] || fail 'provide --role once, with system or vision'
    ROLE=$2
    ROLE_SET=1
    shift 2
done
[[ $ROLE == system || $ROLE == vision ]] || fail "unknown role: $ROLE"
COMMAND=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $ROLE_SET == 1 && $COMMAND != shell && $COMMAND != python ]]; then
    fail '--role is valid only for shell/python; doctor always checks both roles'
fi

# Subprocess-only sanitation: leave the caller's Conda/Agent environment intact.
CLEAN=(-u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD
       -u CMAKE_PREFIX_PATH -u CMAKE_TOOLCHAIN_FILE -u PKG_CONFIG_PATH
       -u PKG_CONFIG_LIBDIR -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH
       -u LIBRARY_PATH -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS
       -u BASH_ENV -u ENV)
while IFS= read -r variable; do
    case "$variable" in CONDA_*|_CONDA_*|_CE_*) CLEAN+=(-u "$variable");; esac
done < <(compgen -e)

native() { /usr/bin/env "${CLEAN[@]}" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$@"; }

case "$COMMAND" in
    help|-h|--help)
        [[ $# == 0 ]] || fail 'help takes no arguments'
        usage
        ;;
    doctor)
        [[ $# == 0 ]] || fail 'doctor takes no arguments and checks both roles'
        cd -- "$ROOT"
        native /usr/bin/python3 -B "$ROOT/scripts/doctor.py"
        ;;
    build|test)
        [[ $# == 0 ]] || fail "$COMMAND takes no arguments"
        if [[ $COMMAND == build ]]; then
            cd -- "$ROOT/yolo_run"
        else
            cd -- "$ROOT/test"
        fi
        native /usr/bin/cmake --preset rk3588-release
        native /usr/bin/cmake --build --preset rk3588-release
        if [[ $COMMAND == test ]]; then
            native /usr/bin/ctest --preset rk3588-release \
                -L cpu
        fi
        ;;
    python|shell)
        PYTHON=/usr/bin/python3
        ROLE_PATH=/usr/bin:/bin
        if [[ $ROLE == vision ]]; then
            PYTHON=$VISION_PREFIX/bin/python
            ROLE_PATH=$VISION_PREFIX/bin:/usr/bin:/bin
        fi
        [[ -x $PYTHON ]] || fail "missing interpreter: $PYTHON; no environment will be created automatically"
        cd -- "$ROOT"
        if [[ $COMMAND == python ]]; then
            if [[ $ROLE == vision ]]; then
                [[ -r /opt/miniforge3/etc/profile.d/conda.sh ]] || fail 'missing Conda shell initialization'
                exec /usr/bin/env "${CLEAN[@]}" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
                    /bin/bash --noprofile --norc -c \
                    'source "$1"; conda activate "$2" || exit "$?"; shift 2; exec "$@"' \
                    yolo-python /opt/miniforge3/etc/profile.d/conda.sh "$VISION_PREFIX" "$PYTHON" "$@"
            fi
            exec /usr/bin/env "${CLEAN[@]}" PATH="$ROLE_PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$@"
        fi
        [[ $# == 0 ]] || fail 'shell takes no arguments'
        [[ $ROLE != vision || -r /opt/miniforge3/etc/profile.d/conda.sh ]] || fail 'missing Conda shell initialization'
        exec /usr/bin/env "${CLEAN[@]}" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
            YOLO_DEV_ROOT="$ROOT" YOLO_DEV_ROLE="$ROLE" \
            /bin/bash --noprofile --rcfile "$ROOT/scripts/dev-shell.bash" -i
        ;;
    *) fail "unknown command: $COMMAND (use --help)";;
esac
