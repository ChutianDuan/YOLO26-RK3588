#!/usr/bin/python3
"""Read-only environment checks; never initialize a model or start a service."""
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VISION = Path('/opt/miniforge3/envs/rk3588-infer')
WHEELS = Path('/home/admin/rk3588-workspace/downloads/wheels')
FAILURES = []


def report(ok, label, detail=''):
    print(f'{"OK" if ok else "FAIL"} {label}' + (f': {detail}' if detail else ''), flush=True)
    if not ok:
        FAILURES.append(label)


def run(argv, *, cwd=None):
    try:
        result = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30, check=False)
        return result.returncode, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def python_command(role, *args):
    return ['/bin/bash', str(ROOT / 'scripts/dev.sh'), '--role', role, 'python', *args]


def python_roles():
    print('Python roles (isolated from user site/PYTHONPATH)', flush=True)
    code = '''import importlib, importlib.util, json, os, sys
modules = sys.argv[1:]
for name in modules:
    importlib.import_module(name)
print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"conda_prefix":os.environ.get("CONDA_PREFIX"),"version":sys.version.split()[0],"imports":modules,"pip":importlib.util.find_spec("pip") is not None}))
'''
    for role, executable, modules, expected in (
        ('system', '/usr/bin/python3', ['json', 'pathlib', 'unittest'], None),
        ('vision', str(VISION / 'bin/python'), ['numpy', 'cv2', 'psutil', 'ruamel.yaml', 'rknnlite.api'], '3.12.15'),
    ):
        status, output = run(python_command(role, '-B', '-c', code, *modules))
        if status:
            report(False, f'{role} Python/imports', output)
            continue
        try:
            info = json.loads(output)
        except json.JSONDecodeError:
            report(False, f'{role} Python/imports', output)
            continue
        report(expected is None or info['version'] == expected, f'{role} Python/imports',
               f"{info['executable']} Python {info['version']}; {', '.join(modules)}")
        report(info['executable'] == executable and
               (info['prefix'] == str(VISION) and info['conda_prefix'] == str(VISION)
                if role == 'vision' else info['conda_prefix'] is None),
               f'{role} interpreter/activation')
        if info['pip']:
            status, output = run(python_command(role, '-B', '-m', 'pip', 'check'))
            report(status == 0, f'{role} pip check', output)
        elif role == 'system':
            print('SKIP system pip check: system Python has no pip; C++ test scripts use the standard library')
        else:
            report(False, 'vision pip check', 'pip is missing')


def locks():
    print('Current environment locks', flush=True)
    specification = ROOT / 'env/environment.yml'
    report(specification.is_file() and 'prefix:' not in specification.read_text(),
           'environment.yml', 'existing rk3588-infer, no machine-specific prefix')
    explicit = ROOT / 'env/conda-linux-aarch64.explicit.lock'
    expected = {}
    for line in explicit.read_text().splitlines():
        if line.startswith('https://'):
            url, _, checksum = line.rpartition('#')
            if not re.fullmatch(r'[0-9a-f]{64}', checksum) or url in expected:
                report(False, 'Conda lock format', line)
                return
            expected[url] = checksum
    records = [json.loads(path.read_text()) for path in (VISION / 'conda-meta').glob('*.json')]
    actual = {record['url']: record['sha256'] for record in records}
    report(bool(expected) and expected == actual, 'Conda lock vs installed metadata', f'{len(records)} packages')

    requirements = ROOT / 'env/requirements-linux-aarch64.lock'
    wanted = {}
    for line in requirements.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})', line)
        if not match:
            report(False, 'pip lock format', line)
            return
        name, version, checksum = match.groups()
        wanted[re.sub(r'[-_.]+', '-', name).lower()] = (version, checksum)
    code = '''import importlib.metadata as m, json, re
print(json.dumps({re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower():d.version for d in m.distributions() if (d.read_text("INSTALLER") or "").strip()=="pip"}))
'''
    status, output = run(python_command('vision', '-B', '-c', code))
    try:
        installed = json.loads(output) if not status else None
    except json.JSONDecodeError:
        installed = None
    report(installed == {name: version for name, (version, _) in wanted.items()} and len(wanted) == 5,
           'pip lock vs installed ownership', 'five pip packages; packaging/setuptools/wheel belong to Conda')
    if WHEELS.is_dir():
        available = {digest(path) for path in WHEELS.glob('*.whl')}
        report(all(checksum in available for _, checksum in wanted.values()), 'local wheel SHA256', str(WHEELS))
    else:
        print(f'SKIP local wheel cache: {WHEELS} is absent; installed environment checks still apply')


def native_tools():
    print('Native C++ toolchain and SDK (no hardware calls)', flush=True)
    report(platform.machine() in ('aarch64', 'arm64'), 'architecture', platform.machine())
    for name, argv in (
        ('G++', ['/usr/bin/g++', '--version']), ('CMake', ['/usr/bin/cmake', '--version']),
        ('Ninja', ['/usr/bin/ninja', '--version']), ('FFmpeg', ['/usr/bin/ffmpeg', '-version']),
        ('OpenCV', ['/usr/bin/pkg-config', '--modversion', 'opencv4']),
        ('FFmpeg libraries', ['/usr/bin/pkg-config', '--modversion', 'libavformat', 'libavcodec', 'libavutil']),
    ):
        status, output = run(argv)
        detail = ', '.join(output.splitlines()) if name == 'FFmpeg libraries' else (output.splitlines()[0] if output else '(no output)')
        report(status == 0, name, detail)
    for path in (
        '/opt/rockchip/cmake/RockchipConfig.cmake',
        '/opt/rknn/runtime-2.5.0/include/rknn_api.h', '/opt/rknn/runtime-2.5.0/lib/librknnrt.so',
        '/opt/rockchip/librga/include/im2d.h', '/opt/rockchip/librga/lib/librga.so',
        '/opt/rockchip/mpp/include/rockchip/rk_mpi.h', '/opt/rockchip/mpp/lib/librockchip_mpp.so',
        '/usr/lib/aarch64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake',
        '/usr/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake',
    ):
        report(Path(path).is_file(), 'SDK/development file', path)
    mediamtx = ROOT / 'tools/mediamtx/v1.21.2/mediamtx'
    if mediamtx.exists() or mediamtx.is_symlink():
        status, output = run([str(mediamtx), '--version'])
        report(status == 0 and '1.21.2' in output, 'MediaMTX version only', output)
    else:
        print('SKIP MediaMTX: optional streaming server is not installed')
    for directory in ('yolo_run', 'test'):
        if directory == 'test' and not (ROOT / directory).exists():
            print('SKIP test native preset: board-local test suite is absent')
            continue
        status, output = run(['/usr/bin/cmake', '--list-presets'], cwd=ROOT / directory)
        report(status == 0 and 'rk3588-release' in output, f'{directory} native preset')
    for name in ('yolo_run', 'camera_run', 'camera_live', 'yolo_multi'):
        executable = ROOT / 'yolo_run/build-rk3588' / name
        if not executable.is_file():
            print(f'SKIP ldd {name}: production binary not built')
            continue
        status, output = run(['/usr/bin/ldd', str(executable)])
        links = dict(re.findall(r'^\s*(\S+) => (\S+)', output, re.MULTILINE))
        expected = {'librknnrt.so': '/opt/rknn/runtime-2.5.0/lib/',
                    'librga.so': '/opt/rockchip/librga/lib/'}
        if name == 'camera_live':
            expected['librockchip_mpp.so.1'] = '/opt/rockchip/mpp/lib/'
        sdk_ok = all(links.get(library, '').startswith(prefix) for library, prefix in expected.items())
        system_ok = all(path.startswith(('/lib/aarch64-linux-gnu/', '/usr/lib/aarch64-linux-gnu/'))
                        for library, path in links.items()
                        if library.startswith(('libopencv_', 'libstdc++', 'libavformat.', 'libavcodec.', 'libavutil.')))
        report(status == 0 and sdk_ok and system_ok and 'not found' not in output and
               'miniforge' not in output.lower() and 'conda' not in output.lower(),
               f'ldd {name}', 'system C++ libraries and existing /opt SDK; no missing/Conda libraries')


def main():
    print(f'YOLO environment doctor: {ROOT}', flush=True)
    python_roles()
    locks()
    native_tools()
    print(f'Result: {len(FAILURES)} failed checks', flush=True)
    return 1 if FAILURES else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'FAIL doctor: {exc}', file=sys.stderr)
        sys.exit(1)
