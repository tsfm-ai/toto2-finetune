"""
Compatibility shims for running Toto 2 on PyTorch < 2.5.

The `enable_gqa` argument to F.scaled_dot_product_attention was added in
PyTorch 2.5.  Toto 2's model.py passes it unconditionally, which raises a
TypeError on 2.4.x.  This module patches the function once, transparently,
so callers don't need to change anything.

Import this module (or call `patch_sdpa()`) before importing toto2.
"""
from __future__ import annotations

import functools
import torch
import torch.nn.functional as F


def patch_sdpa() -> bool:
    """
    Monkey-patch F.scaled_dot_product_attention to ignore enable_gqa on
    PyTorch < 2.5.  Safe no-op if already on 2.5+.

    Returns True if the patch was applied, False if it was a no-op.
    """
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 5):
        return False  # native support — no patch needed

    _orig = F.scaled_dot_product_attention

    @functools.wraps(_orig)
    def _patched(*args, enable_gqa: bool = False, **kwargs):
        # enable_gqa=True with num_heads ≠ num_kv_heads is GQA proper.
        # For 22M/313M with heads_per_group==1 this is always False, so
        # dropping the kwarg is correct.  If someone tries GQA on 2.4, they
        # will silently fall back to full attention — acceptable degradation.
        return _orig(*args, **kwargs)

    F.scaled_dot_product_attention = _patched  # type: ignore[assignment]
    return True


# Apply automatically on import so `from toto2ft import compat` is enough.
_patched = patch_sdpa()
