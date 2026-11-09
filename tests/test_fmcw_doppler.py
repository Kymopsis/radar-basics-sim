import numpy as np
from radar_sim import fmcw as F


def test_doppler_uses_centre_frequency():
    """A folded velocity must land where fc (centre) predicts, not at (fc + B/2)."""
    rad = F.Radar(fc=60.5e9, bw=4e9, tc=100e-6, fs=2.56e6, n_samples=256, n_chirps=64, t_idle=150e-6,
                  noise_dbfs=-60, rx=np.array([[0.0, 0.0]]))
    for v in (7.0, 6.0, 2.0):
        p = (np.abs(F.doppler_fft(F.range_fft(F.cube(rad, [F.Target(r=5.0, v=v)])))) ** 2).sum(1)
        d, rb = np.unravel_index(p.argmax(), p.shape)
        y0, y1, y2 = np.log(p[d - 1, rb]), np.log(p[d, rb]), np.log(p[(d + 1) % p.shape[0], rb])
        vel = (d + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) - rad.n_chirps / 2) * rad.v_res
        exp = (v + rad.v_max) % (2 * rad.v_max) - rad.v_max
        assert abs(vel - exp) < 0.02, (v, vel, exp)
