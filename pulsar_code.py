import time
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from typing import Any, cast, Optional
from scipy.signal import windows

# Optional faster paths
try:
    from numba import njit, prange
    _have_numba = True
except Exception:
    _have_numba = False

try:
    from scipy.ndimage import uniform_filter1d
    _have_uniform_filter = True
except Exception:
    _have_uniform_filter = False


def next_pow2(n):
    """Smallest power of 2 that is >= n."""
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Dedispersion (single-DM shift-and-add, no wraparound)
# ---------------------------------------------------------------------------

def dedisperse_zero_fill(normalized: np.ndarray, sample_shifts: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Shift each channel by its DM delay (in samples) without wraparound.
    More memory-efficient than building large index arrays: copy per-channel
    slices into the output. If `out` is provided it will be filled in-place.
    This is the plain-NumPy version, used when Numba is unavailable, or to
    reconstruct the full 2D dedispersed array for the single winning DM.
    """
    n_time, nchan = normalized.shape
    if out is None:
        dedispersed = np.zeros_like(normalized)
    else:
        dedispersed = out
        dedispersed.fill(0.0)

    for f in range(nchan):
        s = int(sample_shifts[f])
        if s >= 0:
            if s < n_time:
                dedispersed[: n_time - s, f] = normalized[s:, f]
        else:
            s_abs = -s
            if s_abs < n_time:
                dedispersed[s_abs:, f] = normalized[: n_time - s_abs, f]

    return dedispersed


if _have_numba:

    @njit(cache=True)
    def _dedisperse_zero_fill_numba(normalized, sample_shifts, out):
        out.fill(0.0)
        n_time, nchan = normalized.shape
        for f in range(nchan):
            s = int(sample_shifts[f])
            if s >= 0:
                if s < n_time:
                    for t in range(n_time - s):
                        out[t, f] = normalized[t + s, f]
            else:
                s_abs = -s
                if s_abs < n_time:
                    for t in range(n_time - s_abs):
                        out[t + s_abs, f] = normalized[t, f]

    def dedisperse_zero_fill_numba_wrapper(normalized: np.ndarray, sample_shifts: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = np.empty_like(normalized)
        _dedisperse_zero_fill_numba(normalized, sample_shifts, out)
        return out

    # prefer the numba implementation when available
    dedisperse_zero_fill = dedisperse_zero_fill_numba_wrapper


def zero_pad_and_taper(data, taper_alpha=0.1):
    """Taper edges (Tukey window) then zero-pad to the next power of 2,
    since fft.py's radix-2 implementation requires power-of-2 length.
    Returns (padded_data, n_real)."""
    n_real = data.shape[0]
    win = windows.tukey(n_real, alpha=taper_alpha)
    tapered = data * win

    target = next_pow2(n_real)
    if target == n_real:
        return tapered.astype(np.float64), n_real
    padded = np.pad(tapered, (0, target - n_real))
    return padded.astype(np.float64), n_real


# ---------------------------------------------------------------------------
# O(n) median/MAD via quickselect (replaces O(n log n) full sort), plus a
# parallel (prange) DM-trial evaluator that never materializes a full 2D
# dedispersed array for anything but the winning DM.
# ---------------------------------------------------------------------------

if _have_numba:

    @njit(cache=True)
    def _select_kth(arr, k):
        """Return the k-th smallest element of arr (0-indexed), O(n) average.
        Partially reorders arr in place (values <= arr[k] end up left of k,
        values >= arr[k] end up right of k)."""
        lo, hi = 0, arr.shape[0] - 1
        while lo < hi:
            pivot = arr[(lo + hi) // 2]
            i, j = lo, hi
            while i <= j:
                while arr[i] < pivot:
                    i += 1
                while arr[j] > pivot:
                    j -= 1
                if i <= j:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
                    j -= 1
            if k <= j:
                hi = j
            elif k >= i:
                lo = i
            else:
                break
        return arr[k]

    @njit(cache=True)
    def _median_mad(ts):
        """O(n) median and MAD (median absolute deviation) via quickselect,
        instead of the O(n log n) full-sort approach. `ts` is not modified;
        the working copies it creates are what get reordered."""
        n = ts.shape[0]
        work = ts.copy()
        if n % 2 == 1:
            med = _select_kth(work, n // 2)
        else:
            lo = _select_kth(work, n // 2 - 1)
            hi = _select_kth(work, n // 2)
            med = 0.5 * (lo + hi)

        dev = np.abs(ts - med)
        if n % 2 == 1:
            mad = _select_kth(dev, n // 2)
        else:
            lo = _select_kth(dev, n // 2 - 1)
            hi = _select_kth(dev, n // 2)
            mad = 0.5 * (lo + hi)

        if mad == 0.0:
            mad = 1.0
        return med, mad

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_evaluate_dms_snr(normalized_T, sample_shifts_all):
        """Compute the SNR of the dedispersed time series for every DM trial
        in parallel. Only scalars (one SNR per DM) leave the parallel region
        -- no per-DM 2D array is ever materialized, and no per-DM time series
        is kept around beyond what's needed to score that one trial. The
        caller re-derives the full dedispersion for just the winning DM.

        `normalized_T` must be shape (nchan, n_time), C-contiguous -- i.e.
        each channel's samples are contiguous in memory. This is the
        transpose of the original (n_time, nchan) layout. With the old
        layout, `normalized[t + s, f]` for fixed f and varying t strides by
        `nchan` elements (4KB for nchan=512, float64) on every single
        access -- effectively a cache miss per sample, ~n_time*nchan of them
        per DM trial. Indexing along the contiguous axis instead turns that
        into sequential reads, which is the dominant cost for this kernel."""
        nchan, n_time = normalized_T.shape
        n_dm = sample_shifts_all.shape[0]
        snr_arr = np.full(n_dm, -1e99, dtype=normalized_T.dtype)

        for i in prange(n_dm):
            shifts = sample_shifts_all[i]
            time_series = np.zeros(n_time, dtype=normalized_T.dtype)
            max_shift = 0
            for f in range(nchan):
                s = shifts[f]
                if s > max_shift:
                    max_shift = s
                row = normalized_T[f]  # contiguous slice, sequential access below
                if s >= 0:
                    lim = n_time - s
                    for t in range(lim):
                        time_series[t] += row[t + s]
                else:
                    s_abs = -s
                    lim = n_time - s_abs
                    for t in range(lim):
                        time_series[t + s_abs] += row[t]

            trimmed_len = n_time - max_shift if max_shift > 0 else n_time
            if trimmed_len <= 0:
                continue
            trimmed = time_series[:trimmed_len]

            med, mad = _median_mad(trimmed)
            mx = trimmed[0]
            for k in range(1, trimmed_len):
                if trimmed[k] > mx:
                    mx = trimmed[k]
            snr_arr[i] = (mx - med) / mad

        return snr_arr

    @njit(parallel=True, cache=True)
    def _transpose_2d_numba(arr):
        """Parallel, contiguous transpose: parallelizes over channels of the
        *output* so each thread performs one sequential write (reading a
        strided column from the source). Used instead of generic `arr.T` +
        `np.ascontiguousarray(...)`, which on a single core was measured to
        cost nearly as much as the axis=0 reduction it was meant to speed
        up -- with real cores available this scales down further, same as
        the DM-search kernel above."""
        n0, n1 = arr.shape
        out = np.empty((n1, n0), dtype=arr.dtype)
        for i in prange(n1):
            for j in range(n0):
                out[i, j] = arr[j, i]
        return out

    @njit(parallel=True, cache=True)
    def _channel_median_mad_numba(chunk_T):
        """Per-channel median and MAD, one independent quickselect call per
        channel via prange -- replaces two separate `np.median(..., axis=0)`
        calls on the original (n_time, nchan) layout, which strided by
        nchan elements (4KB for nchan=512, float64) on every access and
        profiling showed dominating total runtime (~59% for a 524288x512
        chunk). `chunk_T` must be (nchan, n_time), contiguous per channel
        -- see _transpose_2d_numba."""
        nchan, n_time = chunk_T.shape
        channel_median = np.empty(nchan, dtype=chunk_T.dtype)
        channel_mad = np.empty(nchan, dtype=chunk_T.dtype)
        for f in prange(nchan):
            med, mad = _median_mad(chunk_T[f])
            channel_median[f] = med
            channel_mad[f] = mad
        return channel_median, channel_mad

    @njit(parallel=True, fastmath=True, cache=True)
    def _dedisperse_zero_fill_numba_channel_major(normalized_T, shifts):
        """Same operation as _dedisperse_zero_fill_numba (shift each channel
        by its DM delay, zero-fill, no wraparound), but both input and
        output are (nchan, n_time) -- contiguous per channel -- and channels
        are processed in parallel via prange, since each channel's
        shift-and-copy is independent of every other channel. This replaces
        calling the single-threaded, bad-stride dedisperse_zero_fill on the
        full array for the one-time winning-DM reconstruction."""
        nchan, n_time = normalized_T.shape
        out = np.zeros((nchan, n_time), dtype=normalized_T.dtype)
        for f in prange(nchan):
            s = shifts[f]
            row_in = normalized_T[f]
            row_out = out[f]
            if s >= 0:
                if s < n_time:
                    for t in range(n_time - s):
                        row_out[t] = row_in[t + s]
            else:
                s_abs = -s
                if s_abs < n_time:
                    for t in range(n_time - s_abs):
                        row_out[t + s_abs] = row_in[t]
        return out


def _mp_worker_eval(args):  # pragma: no cover - kept only as a documented no-op stub
    raise RuntimeError(
        "The multiprocessing DM-search path has been removed: parallelism is now "
        "handled inside the compiled kernel via numba prange, which avoids "
        "process-spawn/pickling overhead entirely."
    )


def get_best_time_series(
    file="s120408_215426.sf",
    dm_values=None,
    chunk_length=524288,
    baseline_window=2000,
    taper_alpha=0.1,
    profile: bool = False,
    timing: bool = False,
    use_multiprocessing: bool = False,
    dtype: npt.DTypeLike = np.float64,
):
    """
    Note on `use_multiprocessing`: deprecated and ignored. Parallelism across
    DM trials is now handled inside the compiled kernel via numba `prange`
    (when numba is available), which has much lower overhead than spawning
    threads/processes per call. The parameter is kept only so existing
    callers don't break.

    `dtype`: precision used for the normalized dynamic spectrum and the DM
    search (default float64, matching prior behavior). Passing np.float32
    roughly halves memory bandwidth in the DM-search loop, which is the
    dominant cost for large chunk_length / many DM trials -- worth an A/B
    check against your SNR threshold before relying on it.

    `timing`: if True, print wall-clock time for each stage (FITS I/O,
    unpacking, normalization, DM search, baseline correction, tapering) so
    you can see which stage actually dominates total runtime, rather than
    just the total. Cheap to leave on -- it's a handful of perf_counter()
    calls, not a profiler.

    `profile`: if True, run cProfile over the *entire* function body (not
    just the DM-search step) and print the top 20 calls by cumulative time.
    Use this after `timing` has told you *which* stage is slow, to see
    *which calls within that stage* are slow.
    """
    t_stage = time.perf_counter()
    stage_times = []

    def _checkpoint(label):
        nonlocal t_stage
        now = time.perf_counter()
        stage_times.append((label, now - t_stage))
        t_stage = now

    if profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    # Use memmap when possible to avoid reading the entire file into memory
    with fits.open(file, memmap=True, ignore_missing_simple=True) as hdul:
        subint = cast(fits.BinTableHDU, hdul["SUBINT"])
        header = cast(Any, subint.header)

        # Only read as many rows as needed for the requested chunk_length
        all_raw = cast(Any, subint.data)["DATA"]
        nsblk = int(cast(Any, header["NSBLK"]))
        nrows_available = all_raw.shape[0]
        rows_needed = int(np.ceil(chunk_length / nsblk))
        rows_to_read = min(nrows_available, max(1, rows_needed))
        raw = np.asarray(all_raw[:rows_to_read])
        if timing:
            _checkpoint("FITS I/O (read rows from disk)")
        npol = int(cast(Any, header["NPOL"]))
        nchan = int(cast(Any, header["NCHAN"]))
        tbin = float(cast(Any, header["TBIN"]))
        nbits = int(cast(Any, header.get("NBITS", 4)))
        freqs = np.asarray(cast(Any, subint.data)["DAT_FREQ"][0], dtype=float)

        if raw.size == 0:
            raise ValueError(f"SUBINT.DATA is empty: shape={raw.shape}")
        if freqs.ndim != 1:
            raise ValueError(f"DAT_FREQ must be 1D, but got shape {freqs.shape}")
        if freqs.shape[0] != nchan:
            raise ValueError(f"DAT_FREQ length {freqs.shape[0]} does not match NCHAN {nchan}")
        if nbits != 4:
            raise ValueError(
                f"This unpacking path assumes 4-bit samples, but NBITS={nbits}. "
                "Add a branch for this bit depth before proceeding."
            )

        # Packed bytes: avoid an extra copy where possible by viewing as uint8
        packed = np.asarray(raw, dtype=np.uint8)
        unpacked = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
        # Write directly into the (strided) destination instead of building
        # separate `high`/`low` temporaries -- one allocation instead of three.
        np.right_shift(packed, 4, out=unpacked[..., 0::2])
        np.bitwise_and(packed, 0x0F, out=unpacked[..., 1::2])

        # astropy sometimes can't honor the DATA column's TDIM (e.g. when
        # NBITS < 8 makes the byte repeat count disagree with the TDIM
        # product), and silently hands DATA back as a flat 2D array of
        # (nrows, bytes_per_row) instead of the intended multi-dim shape.
        # Reshape by total sample count (from the header) rather than by
        # unpacked.ndim, so this works whether or not astropy parsed TDIM.
        nrows = raw.shape[0]
        expected_samples = nsblk * npol * nchan
        flat = unpacked.reshape(nrows, -1)
        if flat.shape[1] != expected_samples:
            raise ValueError(
                f"Unpacked DATA has {flat.shape[1]} samples per row, but "
                f"NSBLK*NPOL*NCHAN = {nsblk}*{npol}*{nchan} = {expected_samples}. "
                "Check that NSBLK/NPOL/NCHAN in the header match the DATA column."
            )
        data = flat.reshape(nrows, nsblk, npol, nchan)

        dynamic_spectrum = data[:, :, 0, :].reshape(-1, nchan)

    if timing:
        _checkpoint("4-bit unpack + reshape")

    if dynamic_spectrum.shape[0] == 0:
        raise ValueError("Dynamic spectrum has no time samples")

    chunk = dynamic_spectrum[: min(chunk_length, dynamic_spectrum.shape[0])].astype(dtype)
    # No isfinite/nan_to_num pass here: `chunk` comes straight from unpacking
    # 4-bit unsigned integers (values 0-15), so it is always finite by
    # construction -- that full-array scan was pure overhead.

    if timing:
        _checkpoint("chunk slice + dtype cast")

    if _have_numba:
        chunk_T = _transpose_2d_numba(chunk)
        channel_median, channel_mad = _channel_median_mad_numba(chunk_T)
    else:
        # Plain-NumPy fallback: axis=0 median on the (n_time, nchan) layout.
        # Slower (strided access) but only used when numba isn't available.
        channel_median = np.median(chunk, axis=0)
        channel_mad = np.median(np.abs(chunk - channel_median), axis=0)
        channel_mad[channel_mad == 0] = 1

    centered = chunk - channel_median
    normalized = centered / channel_mad
    zero_dm_signal = np.median(normalized, axis=1, keepdims=True)
    normalized = normalized - zero_dm_signal
    normalized = np.ascontiguousarray(normalized, dtype=dtype)

    if timing:
        _checkpoint("normalization (channel median/MAD, zero-DM subtract)")

    if dm_values is None:
        dm_values = np.arange(12.4, 12.6, 0.1)

    dm_values = np.asarray(dm_values)
    f_ref = np.max(freqs)

    # Precompute delays/shifts for all DMs up front.
    delays = 4.148808e3 * dm_values[:, None] * ((1.0 / freqs**2) - (1.0 / f_ref**2))
    sample_shifts_all = np.rint(delays / tbin).astype(np.int64)

    n_time = normalized.shape[0]
    n_dm = sample_shifts_all.shape[0]

    best_dm = None
    best_time_series = None
    best_dedispersion = None

    use_numba = getattr(get_best_time_series, "use_numba", True)
    if _have_numba and use_numba:
        try:
            # Build the channel-major, contiguous layout once and reuse it
            # for both the SNR search and the winning-DM reconstruction
            # below -- see _transpose_2d_numba / _numba_evaluate_dms_snr
            # docstrings. Only cast dtype if it isn't already float64:
            # `normalized` is float64 by default, so the old unconditional
            # `.astype(np.float64)` was silently making a full redundant
            # 2.1GB copy on every call before the transpose even started.
            normalized_for_kernel = (
                normalized if normalized.dtype == np.float64
                else normalized.astype(np.float64)
            )
            normalized_T = _transpose_2d_numba(normalized_for_kernel)
            snr_arr = _numba_evaluate_dms_snr(normalized_T, sample_shifts_all)
            best_idx = int(np.argmax(snr_arr))
            if snr_arr[best_idx] > -1e98:
                best_dm = float(dm_values[best_idx])
                best_shifts = sample_shifts_all[best_idx]
                # Only the winning DM ever gets a full 2D reconstruction.
                # Reconstruct in channel-major layout too (contiguous read
                # AND write, parallel over channels), then transpose back
                # once at the end -- avoids ever running the old
                # single-threaded, bad-stride dedisperse_zero_fill on the
                # full array, which profiling showed costing several
                # seconds on its own for a single DM trial.
                best_dedispersion_T = _dedisperse_zero_fill_numba_channel_major(
                    normalized_T, best_shifts
                )
                max_shift = int(np.max(best_shifts))
                trimmed_T = (
                    best_dedispersion_T[:, : n_time - max_shift]
                    if max_shift > 0 else best_dedispersion_T
                )
                best_dedispersion = _transpose_2d_numba(trimmed_T)
                if best_dedispersion.dtype != dtype:
                    best_dedispersion = best_dedispersion.astype(dtype)
                best_time_series = np.sum(best_dedispersion, axis=1)
        except Exception:
            # If the numba path fails for any reason, fall back to the
            # plain-Python/NumPy path below.
            best_dm = None
            best_time_series = None
            best_dedispersion = None

    if best_time_series is None:
        # Plain-NumPy sequential fallback (used when numba is unavailable,
        # or if the numba path raised above). No thread pool / process pool:
        # for this fallback simplicity beats micro-parallelism, since the
        # numba+prange path above is the intended fast path.
        best_snr = -np.inf
        dedispersed_buf = np.empty_like(normalized)
        for idx, sample_shifts in enumerate(sample_shifts_all):
            dedispersed = dedisperse_zero_fill(normalized, sample_shifts, out=dedispersed_buf)
            max_shift = int(np.max(sample_shifts))
            trimmed = dedispersed[: n_time - max_shift, :] if max_shift > 0 else dedispersed
            time_series = np.sum(trimmed, axis=1)
            median_ts = np.median(time_series)
            time_series_mad = np.median(np.abs(time_series - median_ts))
            if time_series_mad == 0:
                time_series_mad = 1.0
            snr = (np.max(time_series) - median_ts) / time_series_mad
            if snr > best_snr:
                best_snr = snr
                best_dm = float(dm_values[idx])
                best_time_series = time_series.copy()
                best_dedispersion = trimmed.copy()

    if timing:
        _checkpoint("DM search + winning-DM reconstruction")

    if best_time_series is None or best_dedispersion is None:
        raise RuntimeError("No valid dedispersed time series was found.")

    if baseline_window > 1 and len(best_time_series) > baseline_window:
        signal = best_time_series - np.mean(best_time_series)

        if _have_uniform_filter:
            baseline = uniform_filter1d(signal, size=baseline_window, mode="nearest")
        else:
            pad = baseline_window // 2
            padded_signal = np.pad(signal, (pad, pad), mode="edge")
            cumsum = np.concatenate([[0.0], np.cumsum(padded_signal)])
            ma = (cumsum[baseline_window:] - cumsum[:-baseline_window]) / float(baseline_window)
            baseline = ma[pad: pad + len(signal)]

        best_time_series = signal - baseline

    if timing:
        _checkpoint("baseline correction")

    fft_ready_series, n_real = zero_pad_and_taper(best_time_series, taper_alpha=taper_alpha)

    if timing:
        _checkpoint("taper + zero-pad")
        total = sum(dt for _, dt in stage_times)
        print(f"\n{'stage':45s} {'seconds':>8s}   {'%':>5s}")
        print("-" * 62)
        for label, dt in stage_times:
            pct = 100 * dt / total if total > 0 else 0.0
            print(f"{label:45s} {dt:8.3f}   {pct:4.1f}%")
        print("-" * 62)
        print(f"{'TOTAL':45s} {total:8.3f}\n")

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())

    return fft_ready_series, n_real, best_dedispersion, tbin, freqs, best_dm


def benchmark_get_best_time_series(file="s120408_215426.sf", dm_values=None, trials=3, chunk_length=65536):
    import time

    times = []
    for i in range(trials):
        t0 = time.perf_counter()
        get_best_time_series(file=file, dm_values=dm_values, chunk_length=chunk_length)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"trial {i+1}/{trials}: {dt:.3f}s")

    times = np.array(times)
    print(f"median {np.median(times):.3f}s, mean {np.mean(times):.3f}s, min {np.min(times):.3f}s")
    return times


def _reference_best_time_series_small(normalized, freqs, tbin, dm_values, max_samples=4096):
    """Reference implementation using the original indexing approach for a
    small number of time samples to validate correctness of optimized paths.
    Returns (best_dm, best_time_series).
    """
    n_time = min(normalized.shape[0], max_samples)
    small = normalized[:n_time]
    f_ref = np.max(freqs)
    best_snr = -np.inf
    best = None
    for dm in np.asarray(dm_values):
        delays = 4.148808e3 * dm * ((1.0 / freqs**2) - (1.0 / f_ref**2))
        sample_shifts = np.rint(delays / tbin).astype(int)
        t_idx = np.arange(n_time)[:, None] + sample_shifts[None, :]
        valid = (t_idx >= 0) & (t_idx < n_time)
        clipped_idx = np.clip(t_idx, 0, n_time - 1)
        cols = np.broadcast_to(np.arange(small.shape[1]), t_idx.shape)
        dedispersed = small[clipped_idx, cols]
        dedispersed[~valid] = 0.0
        max_shift = int(np.max(sample_shifts))
        trimmed = dedispersed[: n_time - max_shift] if max_shift > 0 else dedispersed
        time_series = np.sum(trimmed, axis=1)
        median_ts = np.median(time_series)
        time_series_mad = np.median(np.abs(time_series - median_ts))
        if time_series_mad == 0:
            time_series_mad = 1.0
        snr = (np.max(time_series) - median_ts) / time_series_mad
        if snr > best_snr:
            best_snr = snr
            best = (float(dm), time_series)
    return best


if __name__ == "__main__":
    best_ts, n_real, best_dedispersion, tbin, freqs, best_dm = get_best_time_series()
    print("Best DM:", best_dm)
    print("FFT-ready time series length:", len(best_ts))
    print("Real (unpadded, post-trim) samples:", n_real)
    print("Best dedispersed dynamic spectrum shape:", best_dedispersion.shape)