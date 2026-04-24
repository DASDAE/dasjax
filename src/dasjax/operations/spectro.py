"""Spectral patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import dascore as dc
import numpy as np
from dascore.utils.patch import get_dim_axis_value
from scipy.signal import ShortTimeFFT, get_window

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class Fbe(PatchOperation):
    """Compute frequency-band energy with a short-time Fourier transform."""

    overlap: Any = 0
    samples: bool = False
    detrend: bool = False
    taper_window: str | np.ndarray | tuple = "hann"
    fmin: float | None = None
    fmax: float | None = None
    kwargs: dict[str, Any] | None = None
    prepared: dict[str, Any] | None = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def __init__(
        self,
        overlap=0,
        samples=False,
        detrend=False,
        taper_window="hann",
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "detrend", detrend)
        object.__setattr__(self, "taper_window", taper_window)
        object.__setattr__(self, "fmin", fmin)
        object.__setattr__(self, "fmax", fmax)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "prepared", None)
        object.__setattr__(self, "out_boundary", None)
        object.__setattr__(self, "out_coords", None)
        object.__setattr__(self, "out_dtype_codes", None)
        object.__setattr__(self, "out_dims", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        patch = dummy_patch(boundary)
        call_kwargs = dict(self.kwargs or {})
        dim, axis, val = get_dim_axis_value(patch, kwargs=call_kwargs)[0]
        coord = patch.get_coord(dim, require_evenly_sampled=True)
        window_samples = coord.get_sample_count(
            val, samples=self.samples, enforce_lt_coord=True
        )
        step = dc.to_float(coord.step)
        sampling_rate = 1 / abs(step)
        window = (
            self.taper_window
            if isinstance(self.taper_window, np.ndarray)
            else get_window(self.taper_window, window_samples, fftbins=False)
        )
        overlap = (
            coord[:window_samples].get_sample_count(
                self.overlap, samples=self.samples, enforce_lt_coord=True
            )
            if self.overlap is not None
            else 0
        )
        hop = window_samples - overlap
        stft = ShortTimeFFT(
            win=window,
            hop=hop,
            fs=sampling_rate,
            fft_mode="onesided",
            mfft=window_samples,
        )
        frame_times = np.asarray(stft.t(len(coord)))
        frame_starts = (
            np.rint(frame_times * sampling_rate).astype(np.int64) - stft.m_num_mid
        )
        frequencies = np.asarray(stft.f, dtype=np.float64)
        mask = np.ones(len(frequencies), dtype=bool)
        if self.fmin is not None:
            mask &= frequencies >= self.fmin
        if self.fmax is not None:
            mask &= frequencies <= self.fmax
        prepared = {
            "axis": axis,
            "window": np.asarray(window),
            "frame_starts": frame_starts,
            "selected_bins": np.flatnonzero(mask).astype(np.int64),
            "sample_step": step,
            "detrend": self.detrend,
        }
        # Reuse the existing DASCore STFT metadata construction by running on zeros.
        out = patch.stft(
            **{dim: val},
            overlap=self.overlap,
            samples=self.samples,
            detrend=self.detrend,
            taper_window=self.taper_window,
        ).abs()
        ft_dim = next(dim_name for dim_name in out.dims if dim_name.startswith("ft_"))
        if self.fmin is not None or self.fmax is not None:
            out = out.select(**{ft_dim: (self.fmin, self.fmax)})
        out = out.sum(dim=ft_dim, dim_reduce="squeeze")
        out_tree, out_boundary = tree_boundary_from_patch(out)
        return replace(
            self,
            prepared=prepared,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.prepared is not None
        data = kernels.banded_stft_kernel(patch_tree.data, **self.prepared)
        return patch_tree.new(
            data=data,
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary
