"""Chapter 9: range-Doppler map of three moving targets; reads range and velocity of each peak."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, rd_map

o = args(__doc__).parse_args()
r = make("1T2R", n_chirps=128, t_idle=60e-6)
print(r.summary())
tg = [Target(2.0, 0.8), Target(3.5, -1.5, amp=0.6), Target(5.0, 0.0, amp=0.8)]
m, rr, vv, _ = rd_map(r, cube(r, tg))
for t in tg:
    ir = int(round(t.r / r.range_res)); iv = int(np.argmax(m[:, ir]))
    print(f"target r={t.r} v={t.v:+.2f} -> peak at {rr[ir]:.2f} m, {vv[iv]:+.2f} m/s")
if not o.no_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(20 * np.log10(m + 1e-9), origin="lower", aspect="auto", extent=[0, rr[-1], vv[0], vv[-1]])
    ax.set_xlabel("range (m)"); ax.set_ylabel("velocity (m/s)"); show(fig, "ch09_rd.png")
