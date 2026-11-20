"""Create installer/runtime_sources/rocm-<version>-py3-none-any from AMD's rocm sdist.

AMD publishes the `rocm` meta package (the `rocm_sdk` module that torch's ROCm
build imports at start-up to find its HIP libraries) only as an sdist whose
setup.py inspects the build machine. RGB MoCap installs wheels only, so this
script repacks the sdist's pure-Python `rocm_sdk` package as a bundled wheel
source without running setup.py. The dependencies are pinned by the runtime
lock instead (rocm-sdk-core, rocm-sdk-libraries, rocm-sdk-device-<gfx>).

Usage: python -I tools/make_rocm_meta_source.py <rocm-X.Y.Z.tar.gz>
"""
from pathlib import Path
import base64, hashlib, shutil, sys, tarfile

ROOT = Path(__file__).resolve().parents[1]


def record_hash(data):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()


def main(sdist):
    sdist = Path(sdist).resolve()
    with tarfile.open(sdist) as archive:
        top = archive.getmembers()[0].name.split('/')[0]
        version = top.split('-', 1)[1]
        files = {}
        for member in archive.getmembers():
            parts = member.name.split('/')
            if not member.isfile() or len(parts) < 4 or parts[1] != 'src' or parts[2] != 'rocm_sdk':
                continue
            rel = '/'.join(parts[2:])
            if rel.startswith('rocm_sdk/tests/') or '__pycache__' in rel:
                continue
            files[rel] = archive.extractfile(member).read()
        pkg_info = archive.extractfile(top + '/PKG-INFO').read().decode('utf8')
    if 'rocm_sdk/__init__.py' not in files or 'rocm_sdk/_dist_info.py' not in files:
        raise SystemExit('Unexpected rocm sdist layout')
    # Core metadata: name, version, licence and summary from PKG-INFO; the
    # dependency pins live in the runtime lock (installed with --no-deps).
    keep = [line for line in pkg_info.splitlines()
            if line.split(':', 1)[0] in {'Metadata-Version', 'Name', 'Version', 'Summary', 'License', 'Requires-Python', 'Classifier'}]
    keep.append('Requires-Dist: rocm-sdk-core==' + version)
    keep.append('Provides-Extra: libraries')
    keep.append('Requires-Dist: rocm-sdk-libraries==%s; extra == "libraries"' % version)
    dist = 'rocm-%s.dist-info' % version
    files[dist + '/METADATA'] = ('\n'.join(keep) + '\n').encode('utf8')
    files[dist + '/WHEEL'] = b'Wheel-Version: 1.0\nGenerator: rgbmocap-make-rocm-meta-source\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
    files[dist + '/top_level.txt'] = b'rocm_sdk\n'
    files[dist + '/entry_points.txt'] = b'[console_scripts]\nrocm-sdk = rocm_sdk.__main__:main\n'
    record = ['%s,%s,%d' % (name, record_hash(data), len(data)) for name, data in sorted(files.items())]
    record.append(dist + '/RECORD,,')
    files[dist + '/RECORD'] = ('\n'.join(record) + '\n').encode('utf8')
    out = ROOT / 'installer/runtime_sources' / ('rocm-%s-py3-none-any' % version)
    if out.exists():
        shutil.rmtree(out)
    for name, data in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print('Wrote', out, len(files), 'files')


if __name__ == '__main__':
    main(sys.argv[1])
