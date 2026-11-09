"""Chapter 9 - from centimetre range bins to micrometres: range-peak interpolation vs echo phase.

Run:  python examples/ch08_phase_precision.py  [--no-plot]
A static target at 1.5 m is measured in many noisy frames. For each SNR it compares the spread of
  1. the range from the interpolated FFT peak (parabolic interpolation on |X|), and
  2. the displacement from the range-bin phase, dR = lambda_eff * dphi / (4 pi),
with the textbook bound for phase ranging sigma_R = lambda / (4 pi sqrt(2 SNR_bin)).
Textbook point-target model, not any product.
"""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, range_fft, C


def measure(r, snr_db, frames, r0=1.5, seed=0):
    noise_dbfs = -snr_db + 10 * np.log10(r.n_samples)        # SNR per bin = SNR per sample + 10 log10(n_samples)
    rr = make("1T2R", n_chirps=1, noise_dbfs=noise_dbfs)
    k0 = int(round(r0 / rr.range_res))
    rng_pk, ph = [], []
    for i in range(frames):
        x = range_fft(cube(rr, [Target(r=r0)], seed=seed + i, motion=False)[0, 0], win=False)
        m = np.abs(x)
        k = k0 - 2 + int(np.argmax(m[k0 - 2:k0 + 3]))
        a, b, c = m[k - 1], m[k], m[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c)
        rng_pk.append((k + d) * rr.range_res); ph.append(np.angle(x[k0]))
    ph = np.unwrap(np.array(ph))
    return rr, np.std(rng_pk), np.std(ph) * rr.lam / (4 * np.pi)   # fc is the sweep centre


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    r = make("1T2R", n_chirps=1)
    print(f"range bin {r.range_res*100:.2f} cm, wavelength {r.lam*1e3:.2f} mm")
    res = []
    for snr in (10, 20, 30, 40):
        rr, s_pk, s_ph = measure(r, snr, 300)
        bound = r.lam / (4 * np.pi * np.sqrt(2 * 10 ** (snr / 10)))
        res.append((snr, s_pk, s_ph, bound))
        print(f"SNR {snr:2d} dB per bin: peak interpolation std {s_pk*1e3:7.3f} mm, phase std {s_ph*1e6:7.1f} um "
              f"(bound {bound*1e6:6.1f} um)")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        s = np.array(res)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(s[:, 0], s[:, 1] * 1e6, "o-", label="FFT peak interpolation")
        ax.semilogy(s[:, 0], s[:, 2] * 1e6, "s-", label="echo phase")
        ax.semilogy(s[:, 0], s[:, 3] * 1e6, "k--", label="phase bound")
        ax.set_xlabel("SNR per range bin (dB)"); ax.set_ylabel("range std (um)"); ax.legend(); fig.tight_layout()
        show(fig, "ch09_phase_vs_peak.png")
