"""Run GEM-X without importing the base app's installed Python packages."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mocap.gemx_runtime import activate
from mocap.gemx_pipeline import main

configuration = activate()
# AMD fork: keep SOMA on PyTorch; NVIDIA Warp has no AMD backend.
from mocap.soma_rocm import use_pytorch_soma
use_pytorch_soma()
raise SystemExit(main(sys.argv[1:], configuration))
