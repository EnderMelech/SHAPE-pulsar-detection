import numpy as np
from astropy.io import fits
from typing import Any, cast


def get_best_time_series(
    file="fake_pulsar.sf",
    dm_values=None,
    chunk_length=524288,
):
    with fits.open(file, ignore_missing_simple=True) as hdul:
        subint = cast(fits.BinTableHDU, hdul["SUBINT"])
        header = cast(Any, subint.header)

        raw = np.asarray(cast(Any, subint.data)["DATA"])
        nsblk = int(cast(Any, header["NSBLK"]))
        npol = int(cast(Any, header["NPOL"]))
        nchan = int(cast(Any, header["NCHAN"]))
        tbin = float(cast(Any, header["TBIN"]))
        freqs = np.asarray(cast(Any, subint.data)["DAT_FREQ"][0], dtype=float)

        if raw.size == 0:
            raise ValueError(f"SUBINT.DATA is empty: shape={raw.shape}")
        if freqs.ndim != 1:
            raise ValueError(f"DAT_FREQ must be 1D, but got shape {freqs.shape}")
        if freqs.shape[0] != nchan:
            raise ValueError(f"DAT_FREQ length {freqs.shape[0]} does not match NCHAN {nchan}")

        packed = raw.astype(np.uint8)
        high = packed >> 4
        low = packed & 0x0F
        unpacked = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
        unpacked[..., 0::2] = high
        unpacked[..., 1::2] = low

        if unpacked.ndim == 4:
            data = unpacked.reshape(raw.shape[0], nsblk, npol, nchan)
        elif unpacked.ndim == 3:
            data = unpacked.reshape(raw.shape[0], nsblk, 1, nchan)
        else:
            raise ValueError(f"Unexpected packed DATA shape: {raw.shape}")

        dynamic_spectrum = data[:, :, 0, :].reshape(-1, nchan)

    if dynamic_spectrum.shape[0] == 0:
        raise ValueError("Dynamic spectrum has no time samples")

    chunk = dynamic_spectrum[: min(chunk_length, dynamic_spectrum.shape[0])].astype(float)
    channel_median = np.median(chunk, axis=0)
    centered = chunk - channel_median
    channel_mad = np.median(np.abs(centered), axis=0)
    channel_mad[channel_mad == 0] = 1
    normalized = centered / channel_mad
    zero_dm_signal = np.median(normalized, axis=1, keepdims=True)
    normalized = normalized - zero_dm_signal

    if dm_values is None:
        dm_values = np.arange(12.4, 12.6, 0.1)

    best_dm = 0.0
    best_snr = -np.inf
    best_time_series = None
    best_dedispersion = None
    f_ref = np.max(freqs)

    for dm in dm_values:
        delays = 4.148808e3 * dm * ((1.0 / freqs**2) - (1.0 / f_ref**2))
        sample_shifts = np.rint(delays / tbin).astype(int)
        dedispersed = np.zeros_like(normalized)

        for f in range(nchan):
            shift = sample_shifts[f]
            if shift == 0:
                dedispersed[:, f] = normalized[:, f]
            else:
                dedispersed[:, f] = np.roll(normalized[:, f], -shift)

        time_series = np.sum(dedispersed, axis=1)
        median_ts = np.median(time_series)
        time_series_mad = np.median(np.abs(time_series - median_ts))
        if time_series_mad == 0:
            time_series_mad = 1.0

        snr = (np.max(time_series) - median_ts) / time_series_mad
        if snr > best_snr:
            best_snr = snr
            best_dm = dm
            best_time_series = time_series
            best_dedispersion = dedispersed

    if best_time_series is None or best_dedispersion is None:
        raise RuntimeError("No valid dedispersed time series was found.")

    return best_time_series, best_dedispersion, tbin, freqs


if __name__ == "__main__":
    best_ts, best_dedispersion, tbin, freqs = get_best_time_series()
    print("Best dedispersed time series length:", len(best_ts))
    print("Best dedispersed dynamic spectrum shape:", best_dedispersion.shape)
