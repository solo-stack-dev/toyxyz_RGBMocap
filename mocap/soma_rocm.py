"""Keep GEM-X's SOMA body model on PyTorch instead of NVIDIA Warp.

NVIDIA Warp, which py-soma-x uses by default for skinning and skeleton fitting,
has no AMD backend: with PyTorch's ROCm build, ``warp.device_from_torch`` fails
for the Radeon device. SOMA ships dense PyTorch paths for both operations:

- ``SOMALayer(mode=...)``: any mode other than ``"warp"`` uses dense PyTorch
  linear blend skinning;
- ``SkeletonTransfer(use_warp_for_rotations=False)`` uses ``fit_joint_rotations``
  instead of ``fit_rotations_warp``.

``use_pytorch_soma()`` forces both before GEM-X builds its body model. It
changes no installed file and is idempotent.
"""
from __future__ import annotations

_PATCHED = False


def use_pytorch_soma() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import soma.body.soma as soma_body
    import soma.geometry.skeleton_transfer as skeleton_transfer

    layer_init = soma_body.SOMALayer.__init__

    def init_layer(self, *args, **kwargs):
        if kwargs.get('mode', 'warp') == 'warp':
            kwargs['mode'] = 'torch'
        return layer_init(self, *args, **kwargs)

    transfer_init = skeleton_transfer.SkeletonTransfer.__init__

    def init_transfer(self, *args, **kwargs):
        kwargs['use_warp_for_rotations'] = False
        return transfer_init(self, *args, **kwargs)

    soma_body.SOMALayer.__init__ = init_layer
    skeleton_transfer.SkeletonTransfer.__init__ = init_transfer
    _PATCHED = True
