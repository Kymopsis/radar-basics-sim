"""Chapter 15: velocity unfolding with two chirp intervals (Chinese remainder idea) and TDM-MIMO
Doppler compensation with fold hypotheses."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, rd_map, angle_spectrum, C

a = args(__doc__); a.add_argument("--v", type=float, default=9.0, help="true radial velocity (m/s)")
a.add_argument("--az", type=float, default=20.0)
a.add_argument("--vt", type=float, default=30.0, help="true velocity for the two-interval test (m/s)"); o = a.parse_args()

# 1) two chirp intervals
lam = C / 60.5e9; T1, T2 = 50e-6, 60e-6
spans = [lam / (2 * T) for T in (T1, T2)]                    # 2*vmax for each interval
wrap = lambda v, s: (v + s / 2) % s - s / 2
vt = o.vt; m1, m2 = wrap(vt, spans[0]), wrap(vt, spans[1])
c1 = m1 + spans[0] * np.arange(-5, 6); c2 = m2 + spans[1] * np.arange(-5, 6)
i, j = np.unravel_index(np.argmin(np.abs(c1[:, None] - c2[None, :])), (11, 11))
print(f"two intervals: measured {m1:+.2f} and {m2:+.2f} m/s -> unfolded {0.5*(c1[i]+c2[j]):.2f} m/s (true {vt})")

# 2) TDM-MIMO 2T4R: velocity beyond the TDM span spoils the angle until the right fold is used
r = make("2T4R", fc=60.5e9, bw=4e9, tc=40e-6, fs=6.4e6, n_samples=256, n_chirps=64, t_idle=60e-6, noise_dbfs=-30)
cb = cube(r, [Target(3.0, o.v, az=o.az)], seed=3)
m, rr, vv, rd = rd_map(r, cb)
di, ri = np.unravel_index(np.argmax(m), m.shape); vhat = vv[di]
snap = rd[di, :, ri]; pos = r.virtual()[:, 0]; ang = np.linspace(-90, 90, 721)
k = np.repeat(np.arange(len(r.tx)), len(r.rx))
# The angle test only sees the fold modulo n_tx: folds f and f+n_tx give the same phase on every
# transmitter. So test n_tx fold classes, then pick the smallest |v| in the winning class (a tracker
# or a second chirp interval, as in part 1, settles the rest).
ntx = len(r.tx); best = None
for fold in range(ntx):
    v = vhat + fold * 2 * r.v_max
    p = angle_spectrum(snap * np.exp(-1j * 4 * np.pi * v * k * r.t_chirp / r.lam), pos, ang)
    print(f"  fold class {fold} ({v:+6.2f} m/s +- n*{2*ntx*r.v_max:.2f}): peak {20*np.log10(p.max()):6.1f} dB at {ang[np.argmax(p)]:+.1f} deg")
    if best is None or p.max() > best[0]: best = (p.max(), fold, ang[np.argmax(p)])
cands = vhat + (best[1] + ntx * np.arange(-3, 4)) * 2 * r.v_max
vbest = cands[np.argmin(np.abs(cands))]
best = (best[0], vbest, best[2])
print(f"TDM span +-{r.v_max:.2f} m/s, measured {vhat:+.2f} -> chosen {best[1]:+.2f} m/s at {best[2]:+.1f} deg (true {o.v} m/s, {o.az} deg)")
if not o.no_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for v in (vhat, best[1]):
        p = angle_spectrum(snap * np.exp(-1j * 4 * np.pi * v * k * r.t_chirp / r.lam), pos, ang)
        ax.plot(ang, 20 * np.log10(p / p.max()), label=f"compensated with {v:+.1f} m/s")
    ax.set_ylim(-30, 2); ax.set_xlabel("azimuth (deg)"); ax.legend(); fig.tight_layout(); show(fig, "ch15_tdm.png")
