"""Bridges over the DASCore versions dasjax supports.

dasjax declares ``dascore>=0.1.15`` and is developed against the ``dev``
branch, and DASCore has changed three times over that range in ways a
numerical parity layer has to notice. Each difference is asked about rather
than inferred from a version string: a version comparison would have to be
revisited every release, and it cannot describe a development branch that has
some of the changes but not others.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import dascore as dc
import numpy as np


@cache
def normalize_max_uses_absolute() -> bool:
    """Whether ``normalize(norm="max")`` divides by the peak absolute value.

    DASCore's docstring has always described it that way, but the code divided
    by the signed maximum until 0.1.18. The two differ for any slice whose
    most negative sample outweighs its most positive one, so the probe below
    is exactly such a slice.
    """
    probe = dc.Patch(
        data=np.array([[-2.0, 1.0]]),
        coords={"one": np.array([0]), "many": np.arange(2)},
        dims=("one", "many"),
    )
    normalized = np.asarray(probe.normalize("many", norm="max").data)
    # Dividing by 2 lands the peak on -1; dividing by 1 leaves it at -2.
    return bool(np.isclose(normalized[0, 0], -1.0))


@cache
def normalize_skips_nulls() -> bool:
    """Whether the installed DASCore's ``normalize`` ignores NaN.

    DASCore changed this in 0.1.21. Before it, one null blanked every sample
    sharing its slice, and a zero norm zeroed the slice; after it, nulls are
    skipped when the norm is computed, a zero norm leaves its slice alone, and
    the nulls themselves stay null. Both are supported, so the answer is
    measured once against whichever DASCore is installed.
    """
    probe = dc.Patch(
        data=np.array([[np.nan, 3.0, 4.0]]),
        coords={"one": np.array([0]), "many": np.arange(3)},
        dims=("one", "many"),
    )
    normalized = np.asarray(probe.normalize("many", norm="l2").data)
    # The finite samples survive only where the null was left out of the norm.
    return bool(np.isfinite(normalized[0, 1]))


def attrs_carry_coords(attrs: Any) -> bool:
    """Whether this DASCore's ``PatchAttrs`` still holds coordinate metadata.

    DASCore's ``dev`` branch moved coordinate metadata out of the attributes
    entirely, so reading or writing ``attrs.coords`` there is an error rather
    than a no-op.
    """
    return hasattr(attrs, "coords")
