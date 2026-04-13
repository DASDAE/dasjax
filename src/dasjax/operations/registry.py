"""Assembled operation registry."""

from __future__ import annotations

from collections.abc import Iterator

from .basic import OPERATIONS as BASIC_OPERATIONS
from .common import validate_registry
from .filters import OPERATIONS as FILTER_OPERATIONS
from .signal import OPERATIONS as SIGNAL_OPERATIONS
from .spectral import OPERATIONS as SPECTRAL_OPERATIONS
from .types import OperationSpec

OPERATIONS: tuple[OperationSpec, ...] = (
    *BASIC_OPERATIONS,
    *SIGNAL_OPERATIONS,
    *FILTER_OPERATIONS,
    *SPECTRAL_OPERATIONS,
)

validate_registry(OPERATIONS)

_OPERATIONS_BY_NAME: dict[str, OperationSpec] = {operation.name: operation for operation in OPERATIONS}


def get_operation(name: str) -> OperationSpec:
    try:
        return _OPERATIONS_BY_NAME[name]
    except KeyError as exc:
        raise AttributeError(f"Unknown jax patch operation {name!r}.") from exc


def iter_operations() -> Iterator[OperationSpec]:
    return iter(OPERATIONS)


def list_operations() -> tuple[str, ...]:
    return tuple(operation.name for operation in OPERATIONS)
