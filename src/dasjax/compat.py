"""Bridges over the DASCore versions dasjax supports.

dasjax supports the current DASCore release and the ``dev`` branch it is
developed against. Where the two differ, the difference is asked about rather
than inferred from a version string: a version comparison would have to be
revisited every release, and it cannot describe a development branch that has
some of the changes but not others.
"""

from __future__ import annotations

from typing import Any


def attrs_carry_coords(attrs: Any) -> bool:
    """Whether this DASCore's ``PatchAttrs`` still holds coordinate metadata.

    DASCore's ``dev`` branch moved coordinate metadata out of the attributes
    entirely, so reading or writing ``attrs.coords`` there is an error rather
    than a no-op.
    """
    return hasattr(attrs, "coords")
