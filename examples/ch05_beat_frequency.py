"""Chapter 5: one chirp, one target -> beat frequency fb = 2*S*R/c. Solves R from the measured fb."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, range_fft, C

a = args(__doc__); a.add_argument("--range", type=float, default=2.4); o = a.parse_args()
r = make("1T2R", n_chirps=1)
c = cube(r, [Target(r=o.range)])[0, 0]
spec = np.abs(range_fft(c, pad=8))
k = np.argmax(spec[: len(spec) // 2])
fb = k * r.fs / (len(c) * 8)
R = fb * C / (2 * r.slope)
print(f"slope {r.slope/1e12:.1f} MHz/us  beat {fb/1e3:.1f} kHz  -> range {R:.3f} m (true {o.range} m)")
if not o.no_plot:
    import matplotlib.pyplot as plt
    f = np.arange(len(spec)) * r.fs / len(spec) / 1e3
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(np.arange(len(c)) / r.fs * 1e6, c.real); ax[0].set_xlabel("time (us)"); ax[0].set_title("dechirped signal (I)")
    ax[1].plot(f[: len(f) // 2], 20 * np.log10(spec[: len(f) // 2] + 1e-12)); ax[1].set_xlabel("beat frequency (kHz)")
    fig.tight_layout(); show(fig, "ch05_beat.png")
