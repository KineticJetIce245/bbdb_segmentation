#!/usr/bin/env python3
"""Compatibility entry point for the segmentation GUI.

The implementation lives in :mod:`bbdb_seg.app`; use ``python -m bbdb_seg``
or this script, both work.
"""
from bbdb_seg.app import SegmentGui, main

if __name__ == "__main__":
    main()
