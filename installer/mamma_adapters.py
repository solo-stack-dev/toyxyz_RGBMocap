"""Apply RGB MoCap's Windows adapters to a user's pinned upstream sources.

Upstream source and data are downloaded directly, under their own licenses;
they are never included in this repository. Fail on changed upstream syntax.
"""
from pathlib import Path
import ast
import re
import shutil


def replace(path, old, new, count=1):
    source=path.read_text(encoding='utf-8')
    if source.count(old)!=count:raise ValueError('Unsupported upstream source: '+path.name)
    path.write_text(source.replace(old,new),encoding='utf-8')


def adapt_engine(engine, helpers):
    engine=Path(engine)
    path=engine/'landmarks/run_ma_2d.py'
    source=path.read_text(encoding='utf-8');tree=ast.parse(source)
    # Mask-based inference never needs Detectron2 or its extra model download.
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and
             n.module in ('utils.utils_detectron2','detectron2.config')]
    if len(imports)!=2:raise ValueError('Unexpected upstream detector imports.')
    lines=source.splitlines(keepends=True)
    for node in sorted(imports,key=lambda n:n.lineno,reverse=True):del lines[node.lineno-1:node.end_lineno]
    source=''.join(lines);tree=ast.parse(source);edits=[]
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    process=functions['process_data'];args=[a.arg for a in process.args.args]
    expected=['frame_source','detector','device','model','cfg','out_folder','save_cam_output','masks_path']
    if args!=expected+['downsampled_verts_pth']:raise ValueError('Unexpected upstream landmark interface.')
    start=process.body[1].lineno if isinstance(process.body[0],ast.Expr) and isinstance(process.body[0].value,ast.Constant) else process.body[0].lineno
    edits.append((start,process.end_lineno,
        '    from speed_landmarks import process_fast\n    return process_fast('+', '.join(expected)+')\n'))
    main=functions['main']
    first=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg_path' for t in n.targets))
    last=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='detector' for t in n.targets))
    lines=source.splitlines(keepends=True)
    original=''.join(lines[first.lineno-1:last.end_lineno])
    block='    detector = None\n    if masks_folder is None:\n'
    block+=''.join('        '+ast.unparse(n)+'\n' for n in imports)
    block+=''.join('    '+line for line in original.splitlines(keepends=True))
    edits.append((first.lineno,last.end_lineno,block))
    for begin,end,text in sorted(edits,reverse=True):lines[begin-1:end]=[text]
    path.write_text(''.join(lines),encoding='utf-8')
    replace(path,'torch.load(args.weights)',"torch.load(args.weights, weights_only=False, map_location='cpu')")
    for name in ('optimization/scene_debug/renderer.py','optimization/utils/utils_camera.py'):
        p=engine/name
        replace(p,"os.environ['PYOPENGL_PLATFORM'] = 'egl'", "if os.name != 'nt':\n    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')")
    replace(engine/'optimization/utils/fitting.py','    if parallel:',"    if parallel and os.name != 'nt':")
    for name in ('landmarks/utils/video_utils.py','optimization/utils/video_tools.py'):
        replace(engine/name,'"/usr/bin/ffmpeg"',"__import__('imageio_ffmpeg').get_ffmpeg_exe()")
    replace(engine/'optimization/run_ma_3d.py','    use_cam_names = cam_names is not None',
            '    if cam_names:\n        cam_names = sorted(cam_names)\n    use_cam_names = cam_names is not None')
    path=engine/'optimization/utils/optimization.py'
    replace(path,'import torch\n','import torch\nimport os\n')
    # Keep the validated quiet-fit schedule, including its early-stop cadence.
    replace(path,'            show_log = True',"            show_log = iter % 25 == 0")
    source=path.read_text(encoding='utf-8');lines=source.splitlines(keepends=True);tree=ast.parse(source)
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
             ast.unparse(n.value.func)=='self.errors.update']
    if len(matches)!=1:raise ValueError('Unexpected fitting diagnostics.')
    n=matches[0];lines[n.lineno-1:n.end_lineno]=['                        pass  # Scalar diagnostics omitted; uncertainty updates retained.\n']
    path.write_text(''.join(lines),encoding='utf-8')
    for name in ('speed_crop.py','speed_landmarks.py'):shutil.copy2(Path(helpers)/name,engine/name)
    for name in ('landmarks/run_ma_2d.py','optimization/utils/optimization.py'):
        ast.parse((engine/name).read_text(encoding='utf-8'))


def adapt_sdf(root):
    """MSVC integer-width/API fixes; leave the original SDF algorithm intact."""
    root=Path(root)/'src'
    for name,annotation in [('aabb.hpp','__align__(32)'),('triangle.hpp','__align__(48)')]:
        replace(root/name,annotation,'')
    # AMD ROCm: HIP's float2/3/4 types define their own component-wise
    # operators, which make the CUDA-sample vector smoothstep() overloads
    # ambiguous. SDF never calls smoothstep; drop only those three overloads.
    path=root/'helper_math.h';source=path.read_text(encoding='utf-8')
    source,count=re.subn(r'inline __device__ __host__ float([234]) smoothstep\(float\1 a, float\1 b, float\1 x\)\n\{\n.*?\n\}\n','',source,flags=re.S)
    if count!=3:raise ValueError('Unsupported upstream source: helper_math.h')
    path.write_text(source,encoding='utf-8')
    path=root/'bvh_cuda_op.cu';source=path.read_text(encoding='utf-8')
    # The radix-tree split search keeps t=ceil(l/divider)=1 and ends only when
    # `divider *= 2` overflows (signed overflow, undefined). NVIDIA's compiler
    # wraps; AMD's LLVM assumes no overflow and the kernel never finishes.
    # Stop before overflow: the extra t=1 steps of a converged monotone search
    # change nothing, so the tree is the same for any count below 2^29.
    old='for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {'
    if source.count(old)!=1:raise ValueError('Unsupported upstream source: bvh_cuda_op.cu')
    source=source.replace(old,'for (int t = (l + (divider - 1)) / divider; t >= 1 && divider < (1 << 30); divider *= 2) {')
    # Bottom-up bounding boxes: a thread reads the sibling box another thread
    # wrote, ordered only by an atomic counter. AMD GPUs keep per-CU caches
    # that are not coherent, so publish (release) before and acquire after
    # each counter update; NVIDIA needs the same for a guaranteed order.
    for old in ('    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);\n',
                '      curr_counter = atomicAdd(atomic_counters + current_idx, 1);\n'):
        if source.count(old)!=1:raise ValueError('Unsupported upstream source: bvh_cuda_op.cu')
        indent=old[:len(old)-len(old.lstrip())]
        source=source.replace(old,indent+'__threadfence();\n'+old+indent+'__threadfence();\n')
    # Profiler range markers only; HIP on Windows reports them unsupported,
    # which the following error check turns into a failure.
    for call in ('cudaProfilerStart();','cudaProfilerStop();'):
        if source.count(call)!=1:raise ValueError('Unsupported upstream source: bvh_cuda_op.cu')
        source=source.replace(call,'')
    source=source.replace('triangles.type(), "bvh_tree_building"','triangles.scalar_type(), "bvh_tree_building"')
    source=re.sub(r'\blong\b','int64_t',source)
    start=source.index('  AT_DISPATCH_FLOATING_TYPES(');head,tail=source[:start],source[start:]
    lines=[];skip=False
    for line in tail.splitlines(keepends=True):
        if re.match(r'\s*#if (PRINT_TIMINGS|DEBUG_PRINT) == 1',line):
            if skip:raise ValueError('Unexpected nested SDF diagnostics.')
            skip=True;continue
        if skip and re.match(r'\s*#endif',line):skip=False;continue
        if not skip:lines.append(line)
    if skip:raise ValueError('Unterminated SDF diagnostics.')
    # Upstream defines BVH_PROFILING=0, but tests it with #ifdef. Retain that
    # behavior while moving preprocessor directives out of the dispatch macro.
    tail=''.join(lines).replace('#ifdef BVH_PROFILING\n','').replace('#endif\n','')
    path.write_text(head+tail,encoding='utf-8')
