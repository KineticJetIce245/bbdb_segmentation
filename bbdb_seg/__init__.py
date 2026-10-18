"""BBDB cortex segmentation GUI and command-line helpers.

The public ``SegmentGui`` class is loaded lazily so importing a pure submodule
(such as :mod:`bbdb_seg.settings`) does not drag in Tk.
"""
from __future__ import annotations

__all__ = ["SegmentGui"]


def __getattr__(name: str):
    if name == "SegmentGui":
        from .app import SegmentGui
        return SegmentGui
    raise AttributeError(name)
