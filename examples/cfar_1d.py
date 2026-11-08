"""Chapter 14: cell-averaging CFAR on a noisy range profile — threshold follows the noise floor."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, range_fft, ca_cfar_1d

o = args(__doc__).parse_args()
r = make("1T2R", n_chirps=1, noise_dbfs=-40)
s = np.abs(range_fft(cube(r, [Target(1.5, amp=0.2), Target(4.0, amp=0.05)])[0, 0])) ** 2
det, th = ca_cfar_1d(s[: r.n_samples // 2], guard=2, train=10, pfa=1e-3)
x = s[: r.n_samples // 2]
peaks = [i for i in np.where(det)[0] if x[i] >= x[max(i - 1, 0)] and x[i] >= x[min(i + 1, len(x) - 1)]]
print("CFAR cells over threshold:", int(det.sum()), "-> local-max detections at (m):", [round(float(i * r.range_res), 2) for i in peaks])
