"""Windows build adaptations for separately downloaded DroidCalib/LieTorch.

The providers' sources and licenses stay in the private support installation.
These changes adapt compiler/PyTorch interfaces, not the SLAM equations.
"""
from pathlib import Path
import shutil,re

# DroidCalib calls only torch_scatter.scatter_sum and scatter_mean. There is
# no AMD ROCm build of torch_scatter for Windows, and both functions are
# implemented in torch_scatter's own Python layer with Tensor.scatter_add_, so
# this module reproduces that layer (torch_scatter 2.1.2 semantics, MIT,
# Matthias Fey) on plain PyTorch instead of a compiled package.
SCATTER='''"""torch_scatter subset for DroidCalib on AMD ROCm (pure PyTorch).

Same broadcasting, output size and mean/count rules as torch_scatter 2.1.2's
scatter_sum/scatter_add/scatter_mean, which use Tensor.scatter_add_ as well.
"""
import torch

__version__ = '2.1.2+rgbmocap.torch'


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out
'''


def prepare(source,engine):
    source=Path(source);engine=Path(engine)
    # Python modules use absolute imports inside the upstream droid_slam folder.
    shutil.copytree(engine/'pipeline/droidcalib/droid_slam',source/'droid_slam')
    shutil.copytree(engine/'pipeline/droidcalib/src',source/'src')
    for name in ('LICENSE','3rd-party-licenses.txt'):
        shutil.copy2(engine/'pipeline/droidcalib'/name,source/name)
    for path in source.rglob('*'):
        if path.suffix in ('.cpp','.cu'):
            text=path.read_text(encoding='utf8').replace('.type(),','.scalar_type(),')
            if path.parent==source/'src':text=re.sub(r'\blong\b','int64_t',text)
            if path.name=='lietorch_gpu.cu':
                # MSVC otherwise gives CPU templates and __global__ launch
                # stubs identical decorated names; the linker can select the
                # CPU body and dereference a CUDA pointer on the host.
                for name in re.findall(r'__global__ void (\w+)\(',text):
                    text=re.sub(r'\b'+name+r'\b','lietorch_cuda_'+name,text)
                # Dependent member template: nvcc accepts it without the
                # `template` keyword; HIP's clang requires standard C++.
                old='X.Matrix4x4().block<3,3>(0,0)'
                if text.count(old)!=1:raise ValueError('Unsupported upstream source: lietorch_gpu.cu')
                text=text.replace(old,'X.Matrix4x4().template block<3,3>(0,0)')
            path.write_text(text,encoding='utf8')
    masked=(engine/'pipeline/camera/masked_droid_slam.py').read_text(encoding='utf8')
    # Windows worker spawning would duplicate the parent/video/model memory.
    masked=masked.replace('num_workers=8 if os.cpu_count() > 8 else os.cpu_count()', 'num_workers=0')
    # Bound Metric3D activation storage without changing its resolution/precision.
    masked=masked.replace('batch_size = 8','batch_size = 1').replace('batch_size = 32','batch_size = 1')
    (source/'masked_droid_slam.py').write_text(masked,encoding='utf8')
    (source/'torch_scatter').mkdir()
    (source/'torch_scatter/__init__.py').write_text(SCATTER,encoding='utf8')
    (source/'setup_windows.py').write_text('''from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
from pathlib import Path
import os,torch
S=Path(__file__).resolve().parent
os.chdir(S)
# AMD ROCm: Torch hipifies the .cu sources and compiles them with hipcc (MSVC
# runtime selected by Torch); host .cpp files use ROCm clang-cl (CXX).
args={'cxx':['/O2','/MD','/GL-','/wd4819'], 'nvcc':['-O2'] if torch.version.hip else ['-O2','-Xcompiler=/MD','-Xcompiler=/GL-']}
extensions=[CUDAExtension('lietorch_backends',include_dirs=[str(S/'thirdparty/lietorch/lietorch/include'),str(S/'thirdparty/eigen')],sources=['thirdparty/lietorch/lietorch/src/'+n for n in ('lietorch.cpp','lietorch_gpu.cu','lietorch_cpu.cpp')],extra_compile_args=args),
CUDAExtension('droid_backends_intr',include_dirs=[str(S/'thirdparty/eigen')],sources=['src/'+n for n in ('droid.cpp','droid_kernels.cu','correlation_kernels.cu','altcorr_kernel.cu')],extra_compile_args=args)]
for e in extensions:e.library_dirs.append(str(S/'torch-libs'))
setup(name='rgbmocap-prompthmr-camera',version='0.1',ext_modules=extensions,cmdclass={'build_ext':BuildExtension})
''',encoding='utf8')
