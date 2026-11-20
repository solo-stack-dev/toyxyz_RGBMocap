"""Rewrite SOURCE_MANIFEST.json from the files Git tracks or would track.

The manifest lists path, size and SHA-256 for every publishable source file
(tracked plus new, non-ignored files), excluding itself. Run it from any
Python after changing source files:  python -I tools/update_source_manifest.py
"""
from pathlib import Path
import datetime, hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'SOURCE_MANIFEST.json'


def git_files():
    out = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                         cwd=ROOT, capture_output=True, check=True).stdout
    return sorted({p for p in out.decode('utf-8').split('\0') if p and p != MANIFEST.name and (ROOT / p).is_file()})


def main():
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    rows = []
    for name in git_files():
        data = (ROOT / name).read_bytes()
        rows.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    order = {row['path']: i for i, row in enumerate(manifest['files'])}
    rows.sort(key=lambda r: (order.get(r['path'], len(order)), r['path']))
    manifest['files'] = rows
    manifest['date'] = datetime.date.today().isoformat()
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + ('\n' if raw.endswith(b'\n') else '')
    if b'\r\n' in raw:
        text = text.replace('\n', '\r\n')
    MANIFEST.write_bytes(text.encode('utf-8'))
    print('SOURCE_MANIFEST.json:', len(rows), 'files')


if __name__ == '__main__':
    main()
