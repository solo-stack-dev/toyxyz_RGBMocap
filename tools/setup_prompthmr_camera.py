"""Transactional camera extension setup; reuse the verified PromptHMR runtime.

AMD fork: LieTorch and DroidCalib are compiled with HIP through the shared ROCm
build toolchain (installer/rocm_build.py); torch_scatter is the pure-PyTorch
subset written by installer/prompthmr_camera_adapters.py.
"""
from pathlib import Path
import json,os,shutil,sys,uuid,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mocap.amd_gpu import ensure_visible_device;ensure_visible_device(ROOT)  # AMD fork: hide GPUs the ROCm kernels do not cover
from installer import engine,rocm_build
from tools.setup_mamma import compiler,quoted,extract_source,fetch_package
from tools.setup_prompthmr import run
from tools.support_components import download
from tools.session_lock import app_lock
from mocap.storage import atomic_json


def setup(root=ROOT):
    root=Path(root)
    from mocap.prompthmr_runtime import configuration as base_configuration
    from mocap.prompthmr_camera_runtime import identity,inventory,configuration
    base=base_configuration(root);tag=identity(root);py=Path(base['python'])
    with app_lock(root):
        try:
            old=configuration(root,verify=True)
            run([py,'-I','-X','utf8',root/'installer/verify_prompthmr_camera.py','--folder',old['folder'],'--output',Path(old['folder'])/'verification.json'],root/'.setup/logs/prompthmr-camera-verify.log')
            print('Verified PromptHMR camera support retained.',flush=True);return old
        except (OSError,ValueError,RuntimeError,KeyError) as e:print('Preparing PromptHMR camera support: '+str(e),flush=True)
        vc=compiler()
        if shutil.disk_usage(root).free<6*1024**3:raise OSError('Allow 6 GiB free for the camera build, excluding the 1.55 GiB camera models.')
        lock=json.loads((root/'installer/prompthmr-camera-lock.json').read_text(encoding='utf8'))
        folder=root/'.setup/prompthmr-camera-staging'/uuid.uuid4().hex;folder.mkdir(parents=True)
        cache=root/'.setup/prompthmr-camera-downloads';cache.mkdir(exist_ok=True)
        promoted=None;committed=False
        try:
            source=folder/'source';source.mkdir();out=folder/'support';out.mkdir()
            for name,spec in lock['sources'].items():
                archive=download(cache,name+'-'+spec['commit'],spec)
                extract_source(archive,spec,source/'thirdparty'/name)
            from installer.prompthmr_camera_adapters import prepare
            prepare(source,Path(base['engine']))
            # Restore only link libraries from the already pinned Torch archive.
            base_lock=json.loads((root/'installer/prompthmr-runtime-lock.json').read_text(encoding='utf8'))
            wheel=fetch_package(next(p for p in base_lock['packages'] if p['name']=='torch'),root/'.setup/prompthmr-downloads')
            libs=source/'torch-libs';libs.mkdir()
            with zipfile.ZipFile(wheel) as z:
                for name in z.namelist():
                    if name.startswith('torch/lib/') and name.endswith('.lib'):(libs/Path(name).name).write_bytes(z.read(name))
            uv=root/'.setup/bootstrap-git/uv.exe'
            try:python_lib=engine.python_home(root,base_lock['python'])/'libs/python311.lib'
            except (OSError,ValueError,RuntimeError):python_lib=None
            if python_lib is None or not python_lib.is_file():
                run([uv,'python','install',base_lock['python'],'--install-dir',root/'.setup/python','--no-bin','--no-registry','--no-config'],root/'.setup/logs/prompthmr-camera-python.log')
            shutil.copy2(engine.python_home(root,base_lock['python'])/'libs/python311.lib',libs/'python311.lib')
            arch,cap=rocm_build.gpu_target(py,engine.clean_env(root))
            if cap!=base['compute_capability']:raise RuntimeError('This GPU differs from the one PromptHMR was set up for. Run Setup_PromptHMR.cmd again.')
            tools=rocm_build.toolchain(root,uv,py,run)
            print('PromptHMR camera: compiling LieTorch and DroidCalib with HIP for this GPU ('+arch+'). See .setup/logs/prompthmr-camera-build.log.',flush=True)
            build=folder/'build.cmd';build.write_text('\n'.join([
                '@echo off','setlocal DisableDelayedExpansion','chcp 65001 >nul',*rocm_build.build_lines(vc,tools,arch,py.parent/'Lib/site-packages'),
                ' '.join(quoted(x) for x in [py,'-I','-X','utf8',source/'setup_windows.py','build_ext','--build-lib',out,'--build-temp',folder/'build']),
                'exit /b %errorlevel%'])+'\n',encoding='utf8')
            run([os.environ.get('COMSPEC','cmd.exe'),'/d','/c',build],root/'.setup/logs/prompthmr-camera-build.log')
            shutil.copytree(source/'thirdparty/lietorch/lietorch',out/'lietorch',ignore=shutil.ignore_patterns('src','include'))
            shutil.copytree(source/'droid_slam',out/'droid_slam')
            shutil.copytree(source/'torch_scatter',out/'torch_scatter')
            shutil.copy2(source/'masked_droid_slam.py',out/'masked_droid_slam.py')
            for name in ('LICENSE','3rd-party-licenses.txt'):shutil.copy2(source/name,out/name)
            shutil.copy2(source/'thirdparty/lietorch/LICENSE',out/'lietorch/LICENSE')
            run([py,'-I','-X','utf8',root/'installer/verify_prompthmr_camera.py','--folder',out,'--output',out/'verification.json'],root/'.setup/logs/prompthmr-camera-verify.log')
            dest=root/'runtime_prompthmr'/('camera-'+tag[:16]+'-'+uuid.uuid4().hex[:8]);os.replace(out,dest);promoted=dest
            result=json.loads((dest/'verification.json').read_text(encoding='utf8'))
            cfg=dict(version=1,support_id=tag,base_id=base['managed_id'],folder=str(dest.resolve()),compute_capability=result['compute_capability'],gpu_arch=result['gpu_arch'],files=inventory(dest),
                installed_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))
            if identity(root)!=tag or base_configuration(root)!=base:raise RuntimeError('Setup inputs changed. Previous camera support was preserved.')
            atomic_json(root/'.setup/prompthmr-camera-runtime.json',cfg);committed=True
            print('PromptHMR camera support ready: %.1f MiB; prepare the two Moving Camera models in Model Manager.'%(cfg['installed_bytes']/2**20),flush=True)
            return cfg
        finally:
            if folder.exists():engine.remove_owned(root/'.setup/prompthmr-camera-staging',folder)
            if promoted is not None and not committed:engine.remove_owned(root/'runtime_prompthmr',promoted)
