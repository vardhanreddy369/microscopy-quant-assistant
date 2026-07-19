"""Optional segmentation with a pretrained generalist model (Cellpose-SAM).

The classical watershed in :mod:`src.segmentation` assumes object boundaries
from a distance transform. A learned model predicts them instead, which is why
it handles densely packed and touching nuclei far better. On the BBBC039
held-out test split this raises mean F1 across IoU thresholds from 0.770 to
0.870 and cuts merge errors from 258 to 119.

It is optional on purpose:

* it pulls in PyTorch and roughly 1.2 GB of model weights,
* the first run downloads those weights and needs a network,
* it is about a hundred times slower per image than the classical path.

The classical pipeline therefore stays the default, and everything here is
imported lazily so the application runs normally when Cellpose is absent.
"""

from __future__ import annotations

import numpy as np

from .segmentation import SegmentationResult, _drop_small

_MODEL = None


def is_available() -> bool:
    """Whether Cellpose can be imported in this environment."""
    try:
        import cellpose  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def unavailable_reason() -> str:
    """A message explaining why the learned model cannot be used."""
    try:
        import cellpose  # noqa: F401
    except ImportError:
        return (
            "Cellpose is not installed. Install it with "
            "`pip install -r requirements-cellpose.txt` (this pulls in PyTorch, "
            "roughly 1 GB). The classical pipeline needs none of it."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Cellpose is installed but failed to load: {exc}"
    return ""


def load_model(use_gpu: bool = True):
    """Load and cache the pretrained model.

    The first call downloads about 1.2 GB of weights. Afterwards they are read
    from the local cache and no network is needed.
    """
    global _MODEL
    if _MODEL is None:
        from cellpose import models

        _MODEL = models.CellposeModel(gpu=use_gpu)
    return _MODEL


def segment(
    plane: np.ndarray,
    min_size: int = 0,
    use_gpu: bool = True,
    diameter: float | None = None,
) -> SegmentationResult:
    """Segment a prepared analysis plane with the pretrained model.

    Returns the same :class:`SegmentationResult` as the classical path so the
    two are interchangeable everywhere downstream. ``threshold`` is NaN because
    a learned model does not produce one.
    """
    plane = np.asarray(plane, dtype=np.float32)
    if plane.ndim != 2:
        raise ValueError(f"expected a 2-D analysis plane, got shape {plane.shape}")

    model = load_model(use_gpu=use_gpu)

    # Cellpose does its own percentile normalisation, so it is given the
    # prepared plane rather than a re-stretched copy.
    masks, _, _ = model.eval(plane, diameter=diameter)
    labels = np.asarray(masks, dtype=np.int32)

    # The size floor is applied here too, so the control means the same thing
    # whichever segmentation method is selected.
    labels = _drop_small(labels, min_size)

    from skimage.segmentation import relabel_sequential

    labels = relabel_sequential(labels)[0].astype(np.int32)

    return SegmentationResult(
        labels=labels,
        mask=labels > 0,
        threshold=float("nan"),
        n_objects=int(labels.max()),
        method="cellpose",
        separated=True,
    )
