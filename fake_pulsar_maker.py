import numpy as np
from astropy.io import fits


def create_fake_pulsar_sf(filename="fake_pulsar.sf"):
    # Parameters matching your code's expectations
    n_subint = 32
    nsblk = 1024
    npol = 1
    nchan = 128
    tbin = 0.0001  # 0.1 ms sampling time

    # Frequencies in MHz
    freqs = np.linspace(1200.0, 1500.0, nchan)
    f_ref = np.max(freqs)

    total_samples = n_subint * nsblk  # 32,768 samples (2^15)

    # 1. Base 4-bit background noise (values 0 to 15)
    data_4bit = np.random.poisson(lam=6.0, size=(total_samples, nchan)).clip(0, 15).astype(np.uint8)

    # 2. Inject a synthetic pulsar signal at DM = 12.5 pc/cm^3
    dm = 12.5
    period_samples = 2048
    delays = 4.148808e3 * dm * ((1.0 / freqs**2) - (1.0 / f_ref**2))
    sample_shifts = np.rint(delays / tbin).astype(int)

    # Gaussian pulse profile
    pulse_shape = (np.exp(-0.5 * ((np.arange(20) - 10) / 3.0) ** 2) * 8).astype(np.uint8)

    for t_start in range(0, total_samples - period_samples, period_samples):
        for c in range(nchan):
            shift = sample_shifts[c]
            idx = t_start + shift
            if idx + len(pulse_shape) < total_samples:
                data_4bit[idx : idx + len(pulse_shape), c] = np.clip(
                    data_4bit[idx : idx + len(pulse_shape), c] + pulse_shape,
                    0,
                    15,
                )

    # Reshape to (n_subint, nsblk, npol, nchan)
    data_4d = data_4bit.reshape(n_subint, nsblk, npol, nchan)

    # 3. Pack 4-bit nibbles into uint8 bytes
    high_nibbles = data_4d[..., 0::2] << 4
    low_nibbles = data_4d[..., 1::2]
    packed_data = (high_nibbles | low_nibbles).astype(np.uint8)

    # 4. Construct FITS Header & SUBINT Table
    header = fits.Header()
    header["NSBLK"] = nsblk
    header["NPOL"] = npol
    header["NCHAN"] = nchan
    header["TBIN"] = tbin

    col_freq = fits.Column(
        name="DAT_FREQ",
        format=f"{nchan}D",
        array=np.tile(freqs, (n_subint, 1)),
    )

    bytes_per_subint = nsblk * npol * (nchan // 2)
    col_data = fits.Column(
        name="DATA",
        format=f"{bytes_per_subint}B",
        dim=f"({nchan // 2},{npol},{nsblk})",
        array=packed_data.reshape(n_subint, -1),
    )

    subint_hdu = fits.BinTableHDU.from_columns(
        [col_freq, col_data], header=header, name="SUBINT"
    )
    primary_hdu = fits.PrimaryHDU()

    hdul = fits.HDUList([primary_hdu, subint_hdu])
    hdul.writeto(filename, overwrite=True)
    print(f"Created '{filename}' successfully with {total_samples} samples.")


if __name__ == "__main__":
    create_fake_pulsar_sf()