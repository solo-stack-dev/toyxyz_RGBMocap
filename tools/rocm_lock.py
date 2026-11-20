"""Swap a runtime lock's CUDA PyTorch pins for AMD's ROCm Windows build.

This fork runs on AMD Radeon GPUs through PyTorch's ROCm build for Windows
(torch.cuda.* maps to the Radeon card; torch.version.hip is set). Every pinned
runtime keeps its Python version and other packages; only torch/torchvision
and their ROCm SDK wheels change.

Usage:
  python -I tools/rocm_lock.py <lock.json> <wheel-dir> [--group KEY ...]
  python -I tools/rocm_lock.py installer/rocm-devel-lock.json <wheel-dir> --devel

<wheel-dir> holds the downloaded ROCm wheels (see ROCM_PACKAGES). The script
reads each wheel's size and SHA-256 from disk, so the pins are what was
actually downloaded. Groups without torch are left unchanged.
"""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]

ROCM_VERSION = '10.1.0'
TORCH = '2.14.0+rocm' + ROCM_VERSION
TORCHVISION = '0.29.0a0+rocm' + ROCM_VERSION
# GPU target: RDNA3 Navi 31 (RX 7900 XTX/XT/GRE, PRO W7900/W7800).
GFX = 'gfx1100'
INDEX = 'https://stable.repo.amd.com/rocm/whl-next/'


def rocm_packages(py):
    """(name, version, filename, index folder) for one CPython tag such as 'cp310'."""
    tag = '%s-%s-win_amd64' % (py, py)
    return [
        ('torch', TORCH, 'torch-%s-%s.whl' % (TORCH, tag), 'torch'),
        ('torchvision', TORCHVISION, 'torchvision-%s-%s.whl' % (TORCHVISION, tag), 'torchvision'),
        ('amd-torch-device-' + GFX, TORCH, 'amd_torch_device_%s-%s-%s.whl' % (GFX, TORCH, tag), 'amd-torch-device-' + GFX),
        ('amd-torch-device-gfx110x', TORCH, 'amd_torch_device_gfx110x-%s-%s.whl' % (TORCH, tag), 'amd-torch-device-gfx110x'),
        ('amd-torchvision-device-' + GFX, TORCHVISION, 'amd_torchvision_device_%s-%s-%s.whl' % (GFX, TORCHVISION, tag), 'amd-torchvision-device-' + GFX),
        ('rocm-bootstrap', '0.3.0', 'rocm_bootstrap-0.3.0-py3-none-any.whl', 'rocm-bootstrap'),
        ('rocm-sdk-core', ROCM_VERSION, 'rocm_sdk_core-%s-py3-none-win_amd64.whl' % ROCM_VERSION, 'rocm-sdk-core'),
        ('rocm-sdk-libraries', ROCM_VERSION, 'rocm_sdk_libraries-%s-py3-none-win_amd64.whl' % ROCM_VERSION, 'rocm-sdk-libraries'),
        ('rocm-sdk-device-' + GFX, ROCM_VERSION, 'rocm_sdk_device_%s-%s-py3-none-win_amd64.whl' % (GFX, ROCM_VERSION), 'rocm-sdk-device-' + GFX),
    ]


ROCM_META = dict(name='rocm', version=ROCM_VERSION, filename='rocm-%s-py3-none-any.whl' % ROCM_VERSION, url=None,
                 bundled='wheels/rocm-%s-py3-none-any.whl' % ROCM_VERSION,
                 origin="AMD's rocm meta package, published only as an sdist; repacked by tools/make_rocm_meta_source.py without running setup.py")
ROCM_NAMES = {'torch', 'torchvision', 'rocm', 'rocm-bootstrap', 'rocm-sdk-core', 'rocm-sdk-libraries'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def py_tag(filename):
    for part in filename.split('-'):
        if part.startswith('cp3') and part[3:].isdigit():
            return part
    raise ValueError('No CPython tag in ' + filename)


def entry(name, version, filename, folder, wheels):
    path = Path(wheels) / filename
    if not path.is_file():
        raise FileNotFoundError('Download the ROCm wheel first: ' + INDEX + folder + '/' + filename.replace('+', '%2B'))
    return dict(name=name, version=version, filename=filename,
                url=INDEX + folder + '/' + filename.replace('+', '%2B'), sha256=sha(path), bytes=path.stat().st_size)


def convert_packages(packages, wheels):
    """Return the group's package list with CUDA torch replaced by the ROCm set."""
    torch = next((p for p in packages if p['name'] == 'torch'), None)
    if torch is None:
        return packages, False
    py = py_tag(torch['filename'])
    vision = any(p['name'] == 'torchvision' for p in packages)
    keep = [p for p in packages if p['name'] not in ROCM_NAMES and not p['name'].startswith(('amd-torch', 'rocm-sdk-device'))]
    added = [entry(*row, wheels) for row in rocm_packages(py) if vision or 'torchvision' not in row[0]]
    meta = dict(ROCM_META)
    old_meta = next((p for p in packages if p['name'] == 'rocm'), None)
    # The base installer rebuilds bundled wheels deterministically into
    # .setup/bootstrap-git/wheels; optional setups check them by these pins.
    built = ROOT / '.setup/bootstrap-git' / meta['bundled']
    if built.is_file():
        meta.update(sha256=sha(built), bytes=built.stat().st_size)
    else:
        meta.update(sha256=(old_meta or {}).get('sha256', '0' * 64), bytes=(old_meta or {}).get('bytes', 0))
    rows = added + [meta]
    if any('kind' in p for p in packages):
        # MAMMA/PromptHMR locks: every row says how setup installs it.
        for row in rows:
            row['kind'] = 'wheel'
        # Keep their existing (unsorted) order; ROCm rows replace torch in place.
        at = packages.index(torch)
        before = [p for p in packages[:at] if p in keep]
        return before + rows + [p for p in packages[at:] if p in keep], True
    return sorted(keep + rows, key=lambda p: p['name']), True


def devel_lock(wheels):
    """Pins for the shared native-build toolchain (installer/rocm-devel-lock.json).

    The devel wheel holds hipcc, clang-cl and the HIP/ROCm headers as a tarball
    that `rocm-sdk init` expands next to the core/libraries/device wheels it
    hard-links into, so those are pinned with it.
    """
    names = ('rocm-sdk-core', 'rocm-sdk-libraries', 'rocm-sdk-device-' + GFX)
    rows = [entry(*row, wheels) for row in rocm_packages('cp311') if row[0] in names]
    rows.insert(0, entry('rocm-sdk-devel', ROCM_VERSION, 'rocm_sdk_devel-%s-py3-none-win_amd64.whl' % ROCM_VERSION, 'rocm-sdk-devel', wheels))
    meta = dict(ROCM_META)
    built = ROOT / '.setup/bootstrap-git' / meta['bundled']
    meta.update(sha256=sha(built), bytes=built.stat().st_size)
    return dict(version=1, rocm=ROCM_VERSION, gpu_targets=[GFX], packages=rows + [meta])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lock', type=Path)
    parser.add_argument('wheels', type=Path)
    parser.add_argument('--group', action='append', help='Only these group keys (default: every group with torch)')
    parser.add_argument('--devel', action='store_true', help='Write the native-build toolchain lock to <lock> instead')
    args = parser.parse_args()
    if args.devel:
        args.lock.write_text(json.dumps(devel_lock(args.wheels), indent=2) + '\n', encoding='utf8')
        print('Wrote', args.lock)
        return
    lock = json.loads(args.lock.read_text(encoding='utf8'))
    groups = lock['groups'] if 'groups' in lock else [lock]
    changed = []
    for group in groups:
        if args.group and group.get('key') not in args.group:
            continue
        group['packages'], did = convert_packages(group['packages'], args.wheels)
        if did:
            changed.append(group.get('key', args.lock.name))
    if not changed:
        sys.exit('No group with torch was found.')
    if lock.get('shared_cuda_dlls'):
        # CUDA DLL sharing between runtimes; ROCm keeps its libraries in rocm-sdk wheels.
        lock['shared_cuda_dlls'] = []
    if 'download_bytes' in lock and 'groups' in lock:
        # Keep the installer's disk estimates proportional to the new download.
        old = lock['download_bytes']
        unique = {p['sha256']: p['bytes'] for g in lock['groups'] for p in g['packages'] if p.get('url')}
        lock['download_bytes'] = sum(unique.values())
        for key in ('installed_bytes_estimate', 'fresh_free_bytes_required'):
            if key in lock and old:
                lock[key] = max(lock[key], int(lock[key] * lock['download_bytes'] / old))
    if 'registry_sha256' in lock:
        lock['registry_sha256'] = sha(Path(__file__).resolve().parents[1] / 'mocap/runtime_registry.py')
    raw = args.lock.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    with args.lock.open('w', encoding='utf8', newline=newline) as f:
        f.write(json.dumps(lock, indent=2) + ('\n' if raw.endswith(b'\n') else ''))
    print('Converted to ROCm:', ', '.join(changed))


if __name__ == '__main__':
    main()
