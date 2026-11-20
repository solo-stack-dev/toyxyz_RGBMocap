"""Choose the AMD Radeon GPU that the pinned ROCm kernels were built for.

This fork runs on AMD GPUs through PyTorch's ROCm build. ROCm numbers every
AMD GPU in the PC, including the integrated Radeon graphics of Ryzen CPUs, and
that is often device 0. The installed kernels cover one GPU target only (see
tools/rocm_lock.py), so work sent to another device fails with
"CUDA error: invalid kernel file".

ensure_visible_device() runs once at launch and before setup checks. It lists
the GPUs in a throwaway process (so HIP is never initialised with the wrong
device in the caller) and sets HIP_VISIBLE_DEVICES to the best match; every
worker process inherits it. A value the user has already set is respected.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# GPU targets that the locked ROCm device wheels contain kernels for.
SUPPORTED_ARCHES = ('gfx1100',)
VISIBILITY_VARIABLES = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')

_PROBE = r"""
import json, torch
rows = []
if torch.cuda.is_available() and getattr(torch.version, 'hip', None):
    for index in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(index)
        rows.append(dict(index=index, name=p.name, arch=getattr(p, 'gcnArchName', '').split(':')[0],
                         memory=int(p.total_memory), integrated=bool(getattr(p, 'is_integrated', False))))
print(json.dumps(rows))
"""


def list_devices(python: Path) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k.upper() not in VISIBILITY_VARIABLES and not k.upper().startswith('PYTHON')}
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run([str(python), '-I', '-X', 'utf8', '-c', _PROBE], env=env, capture_output=True,
                            text=True, encoding='utf8', errors='replace', timeout=180, creationflags=flags)
    if result.returncode:
        raise RuntimeError('GPU listing failed: ' + (result.stderr or result.stdout).strip()[-600:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def choose(devices: list[dict]) -> dict | None:
    """Best device: a supported target first, then discrete over integrated, then the most memory."""
    if not devices:
        return None
    return max(devices, key=lambda d: (d['arch'] in SUPPORTED_ARCHES, not d['integrated'], d['memory']))


def ensure_visible_device(root: Path | str, python: Path | str | None = None, quiet: bool = False) -> dict | None:
    # MIOpen's default search picks slow convolution algorithms on first use;
    # FAST measured 27x faster on SAM 2's encoder with outputs equal to 3e-6.
    # Workers inherit this from the launcher. A user's own value is kept.
    os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')
    if any(os.environ.get(name) for name in VISIBILITY_VARIABLES):
        return None
    root = Path(root)
    python = Path(python) if python else root / 'runtime' / 'python.exe'
    if not python.is_file():
        return None
    try:
        devices = list_devices(python)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        if not quiet:
            print('GPU selection skipped: ' + str(error), flush=True)
        return None
    best = choose(devices)
    if best is None:
        return None
    if best['arch'] not in SUPPORTED_ARCHES and not quiet:
        print('Warning: no %s GPU found. %s (%s) has no kernels in this ROCm build; GPU work will fail.'
              % ('/'.join(SUPPORTED_ARCHES), best['name'], best['arch']), flush=True)
    if len(devices) > 1:
        os.environ['HIP_VISIBLE_DEVICES'] = str(best['index'])
        if not quiet:
            others = ', '.join('%s (%s)' % (d['name'], d['arch']) for d in devices if d is not best)
            print('Using %s (%s) for AI processing; hidden: %s.' % (best['name'], best['arch'], others), flush=True)
    return best


if __name__ == '__main__':
    chosen = ensure_visible_device(Path(__file__).resolve().parents[1], sys.executable)
    print(json.dumps(dict(chosen=chosen, HIP_VISIBLE_DEVICES=os.environ.get('HIP_VISIBLE_DEVICES'))))
