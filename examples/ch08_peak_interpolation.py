"""Chapter 8: precise range inside a bin — parabolic peak interpolation vs SNR (Monte Carlo)."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, range_fft

o = args(__doc__).parse_args()
rng = np.random.default_rng(0)
for snr_db in [10, 20, 30]:
    errs = []
    for t in range(200):
        r = make("1T2R", n_chirps=1, noise_dbfs=10 * np.log10(256) - snr_db)
        R0 = 2.0 + rng.uniform(0, r.range_res)
        s = np.abs(range_fft(cube(r, [Target(r=R0)], seed=t)[0, 0]))
        k0 = int(R0 / r.range_res); k = k0 - 3 + int(np.argmax(s[k0 - 3:k0 + 4]))  # search near the detection
        a, b, cc = np.log(s[k - 1:k + 2])
        d = 0.5 * (a - cc) / (a - 2 * b + cc)
        errs.append((k + d) * r.range_res - R0)
    print(f"SNR {snr_db} dB (at the FFT peak, before windowing loss)  rms error {np.std(errs)*1e3:.2f} mm  (bin = {r.range_res*1e3:.1f} mm)")
