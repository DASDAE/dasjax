"""Architecture-level checks for the core operation registry."""

from __future__ import annotations

import dasjax
from dasjax import PatchOperation, get_patch_operation, iter_patch_operations


EXPECTED_OPERATIONS = (
    "identity",
    "scale",
    "add",
    "subtract",
    "multiply",
    "divide",
    "maximum",
    "minimum",
    "abs",
    "clip",
    "real",
    "imag",
    "angle",
    "conj",
    "exp",
    "log",
    "log10",
    "log2",
    "is_finite",
    "isinf",
    "isnan",
    "fillna",
    "where",
    "flip",
    "roll",
    "standardize",
    "aggregate",
    "all",
    "any",
    "max",
    "mean",
    "median",
    "min",
    "std",
    "sum",
    "detrend",
    "correlate_shift",
    "dispersion_phase_shift",
    "normalize",
    "differentiate",
    "integrate",
    "taper",
    "taper_range",
    "gaussian_filter",
    "hampel_filter",
    "median_filter",
    "sobel_filter",
    "savgol_filter",
    "notch_filter",
    "wiener_filter",
    "slope_filter",
    "pass_filter",
    "line_mute",
    "slope_mute",
    "pad",
    "hilbert",
    "envelope",
    "phase_weighted_stack",
    "dft",
    "idft",
    "whiten",
    "fbe",
    "velocity_to_strain_rate",
    "velocity_to_strain_rate_edgeless",
    "radians_to_strain",
    "tau_p",
    "correlate",
    "decimate",
    "interpolate",
    "istft",
    "resample",
    "stft",
)

EXPECTED_OPERATION_MODULES = {
    "identity": "dasjax.operations.basic",
    "scale": "dasjax.operations.basic",
    "add": "dasjax.operations.basic",
    "subtract": "dasjax.operations.basic",
    "multiply": "dasjax.operations.basic",
    "divide": "dasjax.operations.basic",
    "maximum": "dasjax.operations.basic",
    "minimum": "dasjax.operations.basic",
    "abs": "dasjax.operations.basic",
    "clip": "dasjax.operations.basic",
    "real": "dasjax.operations.basic",
    "imag": "dasjax.operations.basic",
    "angle": "dasjax.operations.basic",
    "conj": "dasjax.operations.basic",
    "exp": "dasjax.operations.basic",
    "log": "dasjax.operations.basic",
    "log10": "dasjax.operations.basic",
    "log2": "dasjax.operations.basic",
    "is_finite": "dasjax.operations.basic",
    "isinf": "dasjax.operations.basic",
    "isnan": "dasjax.operations.basic",
    "fillna": "dasjax.operations.basic",
    "where": "dasjax.operations.basic",
    "flip": "dasjax.operations.basic",
    "roll": "dasjax.operations.basic",
    "standardize": "dasjax.operations.basic",
    "aggregate": "dasjax.operations.basic",
    "all": "dasjax.operations.basic",
    "any": "dasjax.operations.basic",
    "max": "dasjax.operations.basic",
    "mean": "dasjax.operations.basic",
    "median": "dasjax.operations.basic",
    "min": "dasjax.operations.basic",
    "std": "dasjax.operations.basic",
    "sum": "dasjax.operations.basic",
    "detrend": "dasjax.operations.detrend",
    "correlate_shift": "dasjax.operations.correlate",
    "dispersion_phase_shift": "dasjax.operations.dispersion",
    "normalize": "dasjax.operations.normalize",
    "differentiate": "dasjax.operations.differentiate",
    "integrate": "dasjax.operations.integrate",
    "taper": "dasjax.operations.taper",
    "taper_range": "dasjax.operations.taper",
    "gaussian_filter": "dasjax.operations.filter",
    "hampel_filter": "dasjax.operations.filter",
    "median_filter": "dasjax.operations.filter",
    "notch_filter": "dasjax.operations.filter",
    "pass_filter": "dasjax.operations.filter",
    "savgol_filter": "dasjax.operations.filter",
    "sobel_filter": "dasjax.operations.filter",
    "slope_filter": "dasjax.operations.filter",
    "wiener_filter": "dasjax.operations.filter",
    "line_mute": "dasjax.operations.mute",
    "slope_mute": "dasjax.operations.mute",
    "pad": "dasjax.operations.pad",
    "hilbert": "dasjax.operations.hilbert",
    "envelope": "dasjax.operations.hilbert",
    "phase_weighted_stack": "dasjax.operations.hilbert",
    "dft": "dasjax.operations.fourier",
    "idft": "dasjax.operations.fourier",
    "whiten": "dasjax.operations.whiten",
    "fbe": "dasjax.operations.spectro",
    "velocity_to_strain_rate": "dasjax.operations.strain",
    "velocity_to_strain_rate_edgeless": "dasjax.operations.strain",
    "radians_to_strain": "dasjax.operations.strain",
    "tau_p": "dasjax.operations.taup",
    "correlate": "dasjax.operations.numeric",
    "decimate": "dasjax.operations.numeric",
    "interpolate": "dasjax.operations.numeric",
    "istft": "dasjax.operations.numeric",
    "resample": "dasjax.operations.numeric",
    "stft": "dasjax.operations.numeric",
}


def test_core_operation_names_are_unique() -> None:
    """Ensure each registered operation has a distinct public name."""
    names = [operation.operation_name() for operation in iter_patch_operations()]
    assert len(names) == len(set(names))


def test_all_operations_are_patch_operation_subclasses() -> None:
    """Keep the registry limited to PatchOperation subclasses."""
    for operation in iter_patch_operations():
        assert issubclass(operation, PatchOperation)


def test_list_patch_operations_preserves_registry_order() -> None:
    """Expose operation names in deterministic registration order."""
    assert dasjax.list_patch_operations() == EXPECTED_OPERATIONS


def test_get_patch_operation_resolves_all_registered_names() -> None:
    """Resolve every expected operation through the public lookup helper."""
    for name in EXPECTED_OPERATIONS:
        assert get_patch_operation(name).operation_name() == name


def test_operations_live_in_domain_modules() -> None:
    """Keep operation implementations grouped by DASCore-style domains."""
    for name, module_name in EXPECTED_OPERATION_MODULES.items():
        assert get_patch_operation(name).__module__ == module_name


def test_package_root_exports_only_runtime_surface() -> None:
    """Avoid exporting implementation modules from the package root."""
    assert sorted(dasjax.__all__) == [
        "JaxPatchPipeline",
        "PatchBoundary",
        "PatchOperation",
        "PatchPyTree",
        "get_patch_operation",
        "iter_patch_operations",
        "list_patch_operations",
    ]
