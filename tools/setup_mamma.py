"""Install the optional, isolated MAMMA runtime from pinned upstream files.

Restricted MAMMA/SDF sources are acquired on the user's machine, never bundled.
An existing working generation is retained until installation and GPU checks pass.
AMD fork: native SAM 2/SDF are compiled with HIP through the shared ROCm build
toolchain in installer/rocm_build.py.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
import zipfile
import posixpath
import stat
import time
from pathlib import PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mocap.amd_gpu import ensure_visible_device;ensure_visible_device(ROOT)  # AMD fork: hide GPUs the ROCm kernels do not cover
from installer import engine
from installer.mamma_adapters import adapt_engine,adapt_sdf
from installer import rocm_build
from tools.session_lock import app_lock
from tools.support_components import download,extract
from mocap.storage import atomic_json


def run_step(command,root,cancel,log):
    """Keep opaque setup/build/hash phases visibly active, without fake percentages."""
    started=time.monotonic();notice=started
    with Path(log).open('w',encoding='utf-8') as output:
        process=subprocess.Popen([str(v) for v in command],cwd=root,env=engine.clean_env(root),
            stdout=output,stderr=subprocess.STDOUT,creationflags=0x08000000)
        try:
            while process.poll() is None:
                engine.check_cancel(cancel);time.sleep(.2)
                if time.monotonic()-notice>=15:
                    notice=time.monotonic()
                    print('MAMMA setup: '+Path(log).stem+' is working ('+str(round(notice-started))+' s).',flush=True)
        except BaseException:
            subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/taskkill.exe'),'/PID',str(process.pid),'/T','/F'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=0x08000000)
            process.wait(timeout=30);raise
    if process.returncode:
        raise RuntimeError('A setup step failed. '+Path(log).read_text(encoding='utf-8',errors='replace')[-3500:])


@lru_cache(maxsize=4)
def identity(root=ROOT):
    # Runtime content, not installer presentation, determines numerical identity.
    names=['installer/mamma-runtime-lock.json','installer/mamma_adapters.py',
           'installer/mamma_helpers/speed_crop.py','installer/mamma_helpers/speed_landmarks.py',
           *rocm_build.IDENTITY_FILES]
    return hashlib.sha256(json.dumps({n:engine.sha(Path(root)/n) for n in names},sort_keys=True).encode()).hexdigest()


def compiler():
    """Visual Studio 2022+ (vcvars64.bat); the ROCm compiler is fetched by rocm_build.toolchain()."""
    return rocm_build.visual_studio()


def fetch_package(row,cache):
    # Hash-checked; reuses identical archives from other setups and the bundled rocm wheel.
    return rocm_build.fetch(row,cache,ROOT)



def quoted(value):
    # This function is for a generated batch file, with delayed expansion off.
    value=str(value)
    if any(c in value for c in ('"','\n','\r')):raise ValueError('Unsupported setup path.')
    return '"'+value.replace('%','%%')+'"'


def extract_source(archive_path,spec,destination):
    """Materialize pinned archive-internal aliases as files, never OS links."""
    total=0
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            name=PurePosixPath(entry.filename)
            if '\\' in entry.filename or ':' in entry.filename or name.is_absolute() or '..' in name.parts or name.parts[0]!=spec['archive_root']:
                raise ValueError('Unsafe MAMMA source archive path.')
            if entry.is_dir():continue
            payload=entry
            if stat.S_ISLNK(entry.external_attr>>16):
                target=archive.read(entry).decode('utf-8')
                if '\\' in target or ':' in target or target.startswith('/'):raise ValueError('Unsafe source alias.')
                resolved=PurePosixPath(posixpath.normpath(str(name.parent/target)))
                if resolved.parts[0]!=spec['archive_root'] or '..' in resolved.parts:raise ValueError('Source alias leaves its archive.')
                payload=archive.getinfo(str(resolved))
                if stat.S_ISLNK(payload.external_attr>>16) or payload.is_dir():raise ValueError('Unsupported source alias.')
            total+=payload.file_size
            if total>1024**3:raise ValueError('MAMMA source archive is too large.')
            path=destination/Path(*name.parts[1:]);path.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(payload) as source,path.open('wb') as output:shutil.copyfileobj(source,output,1024**2)


def setup(args):
    lock=json.loads((ROOT/'installer/mamma-runtime-lock.json').read_text(encoding='utf-8'))
    uv=ROOT/'.setup/bootstrap-git/uv.exe'
    if not uv.is_file() or engine.sha(uv)!=lock['uv_sha256']:
        raise RuntimeError('Run install.cmd first to prepare the verified setup helper.')
    with app_lock(ROOT):
        tag=identity();dest=ROOT/'runtime_mamma'/tag[:20]
        configuration=ROOT/'.setup/mamma-runtime.json'
        if configuration.is_file():
            try:
                previous=json.loads(configuration.read_text(encoding='utf-8'))
                candidate=Path(previous['python']).parent
                if previous.get('managed_id')==tag and candidate.is_relative_to(ROOT/'runtime_mamma'):
                    dest=candidate
            except (OSError,ValueError,KeyError,TypeError):pass
        if (dest/'ready.json').is_file():
            try:
                from mocap.mamma_runtime import configure
                report=verify(dest,lock)
                configure(dest/'python.exe',dest/'engine',managed_id=tag)
                print('Verified MAMMA runtime retained. Models and projects were not changed.',flush=True)
                return report
            except Exception as error:print('Existing MAMMA generation needs repair: '+str(error),flush=True)
        vc=compiler()
        if shutil.disk_usage(ROOT).free<18*1024**3:
            raise OSError('MAMMA setup requires approximately 18 GiB free for its runtime, downloads and native build, excluding models and compiler tools.')
        setup=ROOT/'.setup';cache=setup/'mamma-downloads';cache.mkdir(exist_ok=True)
        generation=tag[:20]+'-'+os.urandom(4).hex()
        # Short staging path: ROCm wheels nest up to 152 characters below
        # site-packages and Windows long paths may be disabled (MAX_PATH 260).
        stage=setup/'st'/('mamma-'+generation[-8:]);stage.mkdir(parents=True)
        promoted=None;committed=False
        try:
            logs=setup/'logs';logs.mkdir(exist_ok=True)
            print('MAMMA: separate Python 3.11 / PyTorch ROCm (AMD GPU) environment. Download estimate %.2f GiB; installed estimate 6-8 GiB, excluding models and compiler tools.' % (sum(p['bytes'] for p in lock['packages'])/1024**3),flush=True)
            run_step([uv,'python','install',lock['python'],'--install-dir',setup/'python','--no-bin','--no-registry','--no-config'],ROOT,None,logs/'mamma-python.log')
            shutil.copytree(engine.python_home(ROOT,lock['python']),stage,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            py=stage/'python.exe';target=stage/'Lib/site-packages'
            local={p['name']:fetch_package(p,cache) for p in lock['packages']}
            wheels=[p for p in lock['packages'] if p['kind']=='wheel']
            req=stage/'locked-wheels.txt'
            req.write_text('\n'.join(local[p['name']].as_uri()+' --hash=sha256:'+p['sha256'] for p in wheels)+'\n',encoding='utf-8')
            print('MAMMA: installing hash-verified packages...',flush=True)
            run_step([uv,'pip','install','--python',py,'--target',target,'--no-deps','--offline','--no-index',
                '--require-hashes','--only-binary',':all:','--link-mode','copy','--no-cache','-r',req],ROOT,None,logs/'mamma-packages.log')
            for p in lock['packages']:
                if p['kind']!='source':continue
                print('MAMMA: building pinned Python package '+p['name'],flush=True)
                run_step([uv,'pip','install','--python',py,'--target',target,'--no-deps','--no-build-isolation','--no-index',
                    '--offline','--link-mode','copy',local[p['name']]],ROOT,None,logs/('mamma-'+p['name']+'.log'))
            sources=stage/'sources';sources.mkdir()
            for name,spec in lock['sources'].items():
                archive=download(cache,name+'-'+spec['commit'],spec)
                extract_source(archive,spec,sources/name)
            adapt_engine(sources/'mamma',ROOT/'installer/mamma_helpers')
            adapt_sdf(sources/'sdf')
            os.replace(sources/'mamma',stage/'engine')
            arch,cap=rocm_build.gpu_target(py,engine.clean_env(ROOT))
            tools=rocm_build.toolchain(ROOT,uv,py,lambda command,log:run_step(command,ROOT,None,log))
            build=stage/'build-native.cmd'
            # SAM2_BUILD_ALLOW_ERRORS=0: a failed sam2._C build fails setup.
            lines=['@echo off','setlocal DisableDelayedExpansion','chcp 65001 >nul',*rocm_build.build_lines(vc,tools,arch,target),
                'set "FORCE_CUDA=1"','set "SAM2_BUILD_ALLOW_ERRORS=0"','set "SAM2_BUILD_CUDA=1"']
            for name in ('sam2','sdf'):
                lines+=[' '.join(quoted(x) for x in [uv,'pip','install','--python',py,'--target',target,'--no-deps','--no-build-isolation',
                        '--no-index','--offline','--no-cache','--link-mode','copy',sources/name]),'if errorlevel 1 exit /b 1']
            lines+=['exit /b 0'];build.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            print('MAMMA: compiling native SAM 2 and SDF with HIP for this GPU ('+arch+'). See .setup/logs/mamma-native.log.',flush=True)
            run_step([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',str(build)],ROOT,None,logs/'mamma-native.log')
            engine.prune_runtime(stage)
            report=verify(stage,lock)
            # Keep installations immutable. Publish the runtime pointer only after
            # relocation and a second check; interrupted installs keep the old pointer.
            dest=ROOT/'runtime_mamma'/generation;dest.parent.mkdir(exist_ok=True)
            os.replace(stage,dest)
            promoted=dest
            report=verify(dest,lock)
            from mocap.mamma_runtime import configure
            report.update(runtime_id=tag,gpu_arch=arch,compute_capability=cap,rocm_toolchain=tools['lock'],installed_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))
            atomic_json(dest/'ready.json',report)
            configure(dest/'python.exe',dest/'engine',managed_id=tag)
            committed=True
            try:atomic_json(setup/'mamma-install-report.json',report)
            except OSError:print('MAMMA is ready; the optional disk-size report could not be saved.',flush=True)
            print('MAMMA ready: %.2f GiB installed. Open Model Manager to prepare the model files.' % (report['installed_bytes']/1024**3),flush=True)
            return report
        finally:
            if stage.exists():engine.remove_owned(setup,stage)
            if promoted is not None and not committed:engine.remove_owned(ROOT/'runtime_mamma',promoted)



def verify(folder,lock):
    report=folder/'verification.json'
    run_step([folder/'python.exe','-I','-X','utf8',ROOT/'installer/verify_mamma.py','--lock',ROOT/'installer/mamma-runtime-lock.json',
                '--output',report],ROOT,None,ROOT/'.setup/logs/mamma-verify.log')
    return json.loads(report.read_text(encoding='utf-8'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-licenses',action='store_true',required=True,
        help='I reviewed the MAMMA, pytorch_sdf, SAM 2 and dependency licenses; restricted software is installed locally for permitted use.')
    try:setup(parser.parse_args())
    except Exception as exc:print('MAMMA setup failed: '+str(exc),file=sys.stderr);raise SystemExit(1)
