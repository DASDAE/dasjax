"""Tests for package-level behavior."""

from __future__ import annotations

import importlib
import importlib.metadata

import dasjax


def test_package_version_falls_back_when_metadata_missing(monkeypatch) -> None:
    """Use the unknown version fallback when package metadata is absent."""

    def _missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing_version)
    module = importlib.reload(dasjax)

    assert module.__version__ == "0+unknown"
