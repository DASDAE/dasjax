"""Tests for JAX pytree registration and pipelines."""

import dascore as dc
import jax
import numpy as np
import pytest

from dasjax import JaxPatchPipeline, list_operations
from dasjax.operations import get_operation


def _run_operation(patch, name: str, *args, **kwargs):
    """Apply one internal operation through DASCore's pipe helper."""
    return patch.pipe(get_operation(name).patch_impl, *args, **kwargs)


def test_patch_pytree_roundtrip() -> None:
    patch = dc.get_example_patch()

    leaves, tree = jax.tree_util.tree_flatten(patch)
    out = jax.tree_util.tree_unflatten(tree, leaves)

    assert out.equals(patch)


def test_patch_pytree_restores_datetime_coords() -> None:
    patch = dc.get_example_patch()

    leaves, tree = jax.tree_util.tree_flatten(patch)
    out = jax.tree_util.tree_unflatten(tree, leaves)

    out_time = out.coords.coord_map["time"]
    in_time = patch.coords.coord_map["time"]

    assert out_time.dtype == in_time.dtype
    assert np.array_equal(out_time.values, in_time.values)


def test_patch_tree_map_updates_data_and_coords() -> None:
    patch = dc.get_example_patch()

    out = jax.tree_util.tree_map(lambda x: x + 1, patch)
    out_distance = out.coords.coord_map["distance"]
    in_distance = patch.coords.coord_map["distance"]
    out_time = out.coords.coord_map["time"]
    in_time = patch.coords.coord_map["time"]

    assert np.array_equal(out.data, patch.data + 1)
    assert np.array_equal(out_distance.values, in_distance.values + 1)
    assert np.array_equal(
        out_time.values,
        in_time.values + np.timedelta64(1, "ns"),
    )


def test_pipeline_compile_returns_callable_patch_transform() -> None:
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().identity().scale(2.0).scale(3.0)

    compiled = pipeline.compile()
    out = compiled(patch)

    assert callable(compiled)
    assert np.allclose(out.data, patch.data * 6.0)
    assert out.coords == patch.coords


@pytest.mark.parametrize("operation_name", list_operations())
def test_pipeline_accepts_registered_operations(operation_name: str) -> None:
    pipeline = getattr(JaxPatchPipeline(), operation_name)

    assert callable(pipeline)


def test_pipeline_chains_multiple_operations() -> None:
    patch = dc.get_example_patch()
    pipeline = (
        JaxPatchPipeline()
        .add(1.5)
        .scale(2.0)
        .clip(-0.25, 0.25)
        .detrend(dim="time", type="constant")
    )

    out = pipeline.compile()(patch)
    expected = _run_operation(
        _run_operation(
            _run_operation(
                _run_operation(patch, "add", 1.5),
                "scale",
                2.0,
            ),
            "clip",
            -0.25,
            0.25,
        ),
        "detrend",
        dim="time",
        type="constant",
    )

    assert np.allclose(out.data, expected.data)
    assert out.coords == expected.coords


def test_pipeline_rejects_unknown_operations() -> None:
    pipeline = JaxPatchPipeline()

    try:
        pipeline.mean()
    except AttributeError as exc:
        assert "Unknown jax patch operation" in str(exc)
    else:
        raise AssertionError("Expected an unknown pipeline operation to fail.")
