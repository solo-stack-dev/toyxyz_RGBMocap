"""Shared AMD ROCm toolchain for the optional native (HIP) model extensions.

MAMMA (SAM 2, pytorch_sdf), PromptHMR (SAM 2) and PromptHMR moving camera
(LieTorch, DroidCalib) compile upstream CUDA sources. PyTorch's ROCm build
hipifies them and compiles with AMD's hipcc; host C++ files go through ROCm's
clang-cl because MSVC cl.exe cannot parse the hipified ATen/HIP headers.

The compiler comes from AMD's pinned `rocm-sdk-devel` wheel, installed once into
`.setup/rocm-devel/<lock id>/` and expanded there with the `rocm` package's own
`rocm_sdk` expander (the same thing `rocm-sdk init` does). It is a build tool
only: never inside a verified runtime, because expansion rewrites the devel
wheel's RECORD and deletes its tarball. Visual Studio 2022 or newer supplies
the linker, the MSVC runtime headers and the Windows SDK.
"""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess

from . import engine

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'installer/rocm-devel-lock.json'
COMPAT_HEADER = ROOT / 'installer/rocm/hip_cuda_compat.h'
# Build inputs that change compiled modules; part of each runtime's identity.
IDENTITY_FILES = ('installer/rocm_build.py', 'installer/rocm-devel-lock.json', 'installer/rocm/hip_cuda_compat.h',
                  'installer/rocm/include/device_launch_parameters.h')
HOSTS = {'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org', 'stable.repo.amd.com'}
# Expanded devel tree plus its hard-linked sibling wheels, measured 4.9 GiB.
TOOLCHAIN_BYTES = 6 * 1024 ** 3
VS_HELP = ('Install Visual Studio 2022 or newer (Build Tools or Community) with "Desktop development with C++", '
           'the MSVC x64 compiler and a Windows SDK.')


def visual_studio():
    """vcvars64.bat of the newest Visual Studio 2022+ with the x64 C++ tools."""
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():
        raise RuntimeError(VS_HELP)
    found = subprocess.run([str(vswhere), '-utf8', '-latest', '-products', '*', '-version', '[17.0,)',
                            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    path = found.stdout.strip().splitlines()
    vc = Path(path[0]) / 'VC/Auxiliary/Build/vcvars64.bat' if path else None
    if vc is None or not vc.is_file():
        raise RuntimeError('No Visual Studio C++ x64 compiler was found. ' + VS_HELP)
    kits = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Windows Kits/10/Include'
    if not kits.is_dir() or not any((p / 'ucrt').is_dir() for p in kits.iterdir()):
        raise RuntimeError('The Windows SDK was not found. ' + VS_HELP)
    return vc


def lock():
    return json.loads(LOCK.read_text(encoding='utf-8'))


def _valid(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and engine.sha(path) == row['sha256']


def _reuse(target, row, root):
    """Link an identical verified archive another setup already downloaded."""
    setup = Path(root) / '.setup'
    candidates = [setup / 'rocm-wheels' / row['filename']]
    candidates += [setup / name / row['sha256'] / row['filename'] for name in
                   ('downloads', 'mamma-downloads', 'prompthmr-downloads', 'gemx-downloads', 'rocm-devel/downloads')]
    for source in candidates:
        if source == target or not source.is_file() or source.stat().st_size != row['bytes']:
            continue
        if engine.sha(source) != row['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return True
    return False


def fetch(row, cache, root=ROOT):
    """Hash-checked package archive: bundled, reused from another cache, or downloaded."""
    target = Path(cache) / row['sha256'] / row['filename']
    if _valid(target, row):
        return target
    if row.get('bundled'):
        # Bundled wheels are rebuilt deterministically by install.cmd.
        source = engine.inside(Path(root) / '.setup/bootstrap-git', Path(root) / '.setup/bootstrap-git' / row['bundled'])
        if not _valid(source, row):
            raise RuntimeError('The bundled ' + row['filename'] + ' is missing or changed. Run install.cmd (or update.cmd) first.')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
    if _reuse(target, row, root) and _valid(target, row):
        return target
    from tools.pinned_download import download
    return download(Path(cache) / row['sha256'], row['name'], row, filename=row['filename'],
                    allowed_hosts=HOSTS, max_seconds=24 * 60 * 60)


def toolchain(root, uv, python, run):
    """Install (once) and return the ROCm build toolchain paths.

    run(command, log_path) executes a step and raises on failure.
    """
    root = Path(root)
    pins = lock()
    ident = engine.sha(LOCK)[:12]
    base = root / '.setup/rocm-devel'
    folder = base / ident
    site = folder / 'site'
    info = dict(lock=ident, rocm=pins['rocm'], rocm_home=str(site / '_rocm_sdk_devel'),
                hipcc=str(site / '_rocm_sdk_devel/bin/hipcc.exe'),
                clang_cl=str(site / '_rocm_sdk_devel/lib/llvm/bin/clang-cl.exe'),
                compat_header=str(COMPAT_HEADER))
    ready = folder / 'ready.json'
    try:
        if json.loads(ready.read_text(encoding='utf-8')) == info and all(Path(info[k]).is_file() for k in ('hipcc', 'clang_cl')):
            return info
    except (OSError, ValueError):
        pass
    for old in (base.iterdir() if base.is_dir() else ()):
        # One toolchain generation at a time; downloads are kept for reuse.
        if old.is_dir() and old.name != 'downloads':
            engine.remove_owned(base, old)
    needed = TOOLCHAIN_BYTES + sum(p['bytes'] for p in pins['packages'])
    if shutil.disk_usage(root).free < needed:
        raise OSError('The ROCm build toolchain needs about %.1f GiB free (installed once in .setup/rocm-devel).' % (needed / 1024 ** 3))
    folder.mkdir(parents=True)
    logs = root / '.setup/logs'
    logs.mkdir(parents=True, exist_ok=True)
    print('ROCm build toolchain: preparing AMD ROCm %s compiler (hipcc/clang) in .setup/rocm-devel, about 5 GiB, once for all optional setups.' % pins['rocm'], flush=True)
    cache = base / 'downloads'
    files = [fetch(p, cache, root) for p in pins['packages']]
    req = folder / 'wheels.txt'
    req.write_text('\n'.join(f.as_uri() + ' --hash=sha256:' + p['sha256'] for f, p in zip(files, pins['packages'])) + '\n', encoding='utf-8')
    run([uv, 'pip', 'install', '--python', python, '--target', site, '--no-deps', '--offline', '--no-index',
         '--require-hashes', '--only-binary', ':all:', '--link-mode', 'copy', '--no-cache', '-r', req], logs / 'rocm-devel-install.log')
    # Same expansion as `rocm-sdk init`, scoped to this folder only.
    code = ('import sys;sys.path.insert(0,sys.argv[1]);from rocm_sdk import _devel;'
            'print(_devel.get_devel_root())')
    run([python, '-I', '-X', 'utf8', '-B', '-c', code, site], logs / 'rocm-devel-expand.log')
    if not all(Path(info[k]).is_file() for k in ('hipcc', 'clang_cl')):
        raise RuntimeError('The ROCm devel toolchain did not expand completely. See .setup/logs/rocm-devel-expand.log.')
    engine.atomic_json(ready, info)
    return info


def gpu_target(python, env=None):
    """(gfx arch, [major, minor]) of the visible ROCm GPU, from the target interpreter."""
    code = ("import json,torch;assert torch.cuda.is_available() and torch.version.hip,'No ROCm GPU';"
            "p=torch.cuda.get_device_properties(0);"
            "print(json.dumps(dict(arch=p.gcnArchName.split(':')[0],capability=list(torch.cuda.get_device_capability()),name=p.name,hip=torch.version.hip)))")
    result = subprocess.run([str(python), '-I', '-X', 'utf8', '-B', '-c', code], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', env=env, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('The AMD GPU could not be used from the new runtime: ' + (result.stderr or result.stdout)[-1500:])
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if not re.fullmatch(r'gfx[0-9a-f]{3,5}', info['arch']):
        raise ValueError('Invalid AMD GPU architecture: ' + info['arch'])
    supported = lock()['gpu_targets']
    if info['arch'] not in supported:
        raise RuntimeError('%s (%s) is not a supported GPU target of this ROCm build (%s).' % (info['name'], info['arch'], ', '.join(supported)))
    return info['arch'], info['capability']


def quoted(value):
    # For generated batch files, with delayed expansion off.
    value = str(value)
    if any(c in value for c in ('"', '\n', '\r')):
        raise ValueError('Unsupported setup path.')
    return '"' + value.replace('%', '%%') + '"'


def build_lines(vc, tools, arch, site_packages, jobs=4):
    """Batch-file lines that prepare a HIP extension build (vcvars + ROCm)."""
    ninja = next(iter(Path(site_packages).glob('ninja/data/bin/ninja.exe')), None) or next(iter(Path(site_packages).glob('bin/ninja.exe')), None)
    if ninja is None:
        raise RuntimeError('The pinned ninja build tool is missing from the runtime.')
    header = tools['compat_header'].replace('\\', '/')
    if ' ' in header:
        raise RuntimeError('Move the app to a folder without spaces in its path to build AMD GPU extensions.')

    def setvar(name, value):
        return 'set "' + name + '=' + str(value).replace('%', '%%') + '"'
    return ['call ' + quoted(vc), 'if errorlevel 1 exit /b 1',
            # Torch uses ninja and the CXX host compiler only when ninja is on PATH.
            'set "PATH=' + str(ninja.parent).replace('%', '%%') + ';%PATH%"',
            # Not ROCM_PATH: clang then looks for amdgcn bitcode in the wrong place.
            setvar('ROCM_HOME', tools['rocm_home']), setvar('HIP_PATH', tools['rocm_home']), 'set "ROCM_PATH="',
            setvar('CXX', tools['clang_cl']), setvar('PYTORCH_ROCM_ARCH', arch),
            setvar('MAX_JOBS', jobs), 'set "DISTUTILS_USE_SDK=1"', 'set "MSSdk=1"', 'set "VSLANG=1033"',
            # HIP 7 requires 64-bit warp masks: widen the CUDA 32-bit idiom.
            # installer/rocm/include stands in for CUDA-only headers.
            setvar('HIPCC_COMPILE_FLAGS_APPEND', '-include ' + header + ' -I' + header.rsplit('/', 1)[0] + '/include')]
