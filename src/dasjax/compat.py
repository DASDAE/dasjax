"""Bridges over the DASCore versions dasjax supports.

dasjax supports the current DASCore release and the ``dev`` branch it is
developed against. Where the two differ, the difference is asked about rather
than inferred from a version string: a version comparison would have to be
revisited every release, and it cannot describe a development branch that has
some of the changes but not others.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import get_window

try:  # DASCore moved the taper edge into a shared helper on the dev branch.
    from dascore.utils.signal import get_ramp as _get_ramp
except ImportError:  # The release dasjax also supports has no such helper.
    _get_ramp = None


# DASCore's two own window names, each the scipy window it means. Newer
# DASCore resolves these itself; the release dasjax also supports knows them
# only through a table of its own, so the fallback has to say them here.
_WINDOW_ALIASES = {"cos": "hann", "ramp": "triang"}


def _symmetric_window(window_type: Any, size: int) -> np.ndarray:
    """Return a symmetric window, the shape DASCore tapers were cut from."""
    if isinstance(window_type, str):
        window_type = _WINDOW_ALIASES.get(window_type, window_type)
    return np.asarray(get_window(window_type, size, fftbins=False), dtype=np.float64)


def attrs_carry_coords(attrs: Any) -> bool:
    """Whether this DASCore's ``PatchAttrs`` still holds coordinate metadata.

    DASCore's ``dev`` branch moved coordinate metadata out of the attributes
    entirely, so reading or writing ``attrs.coords`` there is an error rather
    than a no-op.
    """
    return hasattr(attrs, "coords")


def taper_ramp(window_type: Any, length: int) -> np.ndarray:
    """Return the rising edge ``Patch.taper`` climbs over ``length`` samples.

    Parameters
    ----------
    window_type
        The window whose edge this is; anything ``scipy.signal.get_window``
        accepts.
    length
        How many samples the ramp spans.
    """
    if _get_ramp is not None:
        return np.asarray(_get_ramp(window_type, length), dtype=np.float64)
    # Before ``get_ramp``, DASCore's taper took the first half of a symmetric
    # window of twice the ramp's length.
    return _symmetric_window(window_type, 2 * length)[:length]


def taper_range_ramp(window_type: Any, length: int) -> np.ndarray:
    """Return the rising edge ``Patch.taper_range`` climbs over ``length`` samples.

    Parameters
    ----------
    window_type
        The window whose edge this is; anything ``scipy.signal.get_window``
        accepts.
    length
        How many samples the ramp spans.
    """
    if _get_ramp is not None:
        return np.asarray(_get_ramp(window_type, length), dtype=np.float64)
    # Before ``get_ramp``, taper_range built its edge from a symmetric window
    # one sample longer than twice the ramp, so the peak fell just past it.
    return _symmetric_window(window_type, 2 * length + 1)[:length]
