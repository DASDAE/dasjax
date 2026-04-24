"""
Core modules for DASJax
"""
from typing import ClassVar

import dascore as dc
import jax
import numpy as np
from pyparsing.tools.cvt_pyparsing_pep8_names import camel_to_snake

COORDTYPE_CODES = {
    1: np.float32,
    2: np.float64,
    3: np.int32,
    4: np.int64,
    5: 'datetime64[ns]',
    6: 'timedelta64[ns]',
}


class PatchPyTree:
    """
    A Dataclass for representing Patches as pytrees.

    Patches get automatically converted into pytrees when entering a pipeline.
    They are rebuilt on the other end.

    Parameters
    ----------
    data (dynamic)
        The array data.
    dims (static)
        A tuple of dims.
    static_coords (static)
        Any string coordinates.
    dynamic_coords (dynamic)
        Any numeric coords. Note: datetime64 and timedelta64 are converted
        to ints and marked in the coord_type dict.
    static_attrs (static)
        Static attributes. Note: these will only be populated with values
        requested by the pipeline.
    dynamic_attrs (dynamic)
        Dynamic attributes. Note: these will only be populated with values.
    coord_types (dynamic):
        The name and type of coordinate. This is mostly for re-assembling
        the patch.
    """
    data: jax.ndarray
    dims: tuple[str, ...]
    static_coords: dict[str, jax.ndarray]
    dynamic_coords: dict[str, jax.ndarray]
    static_attrs: dict[str, int | float | str]
    dynamic_attrs: dict[str, int | float]
    coord_types: dict[str, int]



class PatchOperationBase:
    """
    A jax-accelerated patch operation for DASCore.

    To implement a Patch Operation, subclass this then define the required
    parameters (data-class style). Next, define the required attrs as either
    dynamic or static. Then define the data_func and/or metadata_func. These
    normally run in parallel with the data_func compling to fused jax kernels.

    Parameters
    ----------
    static_attrs
        A tuple of the attrs that should be treated as static. These should
        not change frequently from patch to patch as each new value can
        trigger a re-compile.

    dynamic_attrs
        A tuple of attrs that should be treated as dynamic. These should be
        strictly numeric but can very from patch to patch without triggering
        recompilation.
    """
    # Parameters go here.

    # Class level
    static_attrs: ClassVar[tuple[str, ...]] = ()
    dynamic_attrs: ClassVar[tuple[str, ...]] = ()

    @property
    def pipe_method_name(self):
        """
        The name which will be added as a method on DASJax pipe.

        Can be overwritten by subclasses as simple parameter.
        """
        name = self.__name__
        return camel_to_snake(name)  # Need to define this.


    def data_func(self, pytree: PatchPyTree) -> PatchPyTree:
        """
        A function that is applied primarily on the patch data
        (or sometimes coord) values.
        """
        return pytree

    def metadata_func(
            self, attrs: dc.PatchAttrs, coords: dc.PatchCoords
    ) -> tuple[dc.PatchAttrs, dc.PatchCoords]:
        """
        A function that updates the metadata (coords, attrs) for a patch.
        """
        return attrs, coords

    def cross_metadata(
            self, pytree: PatchPyTree, attrs: dc.PatchAttrs, coords: dc.PatchCoords
    ) -> tuple[PatchPyTree, dc.PatchAttrs, dc.PatchCoords]:
        """
        A function that allows the metadata to be updated from a pytree.

        This should only be defined by a subclass when necessary as it will
        break the complication chain (eg each PatchOperation with cross_metadata
        must be at the end of a compiled chain).
        """


