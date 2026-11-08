"""Chapter 6: two targets closer and closer; when do they merge? Resolution c/2B."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, range_fft

o = args(__doc__).parse_args()
r = make("1T2R", n_chirps=1)
print(f"range resolution c/2B_eff = {r.range_res*100:.2f} cm")
res = []
for gap in [2.0, 1.5, 1.0, 0.8, 0.5]:
    c = cube(r, [Target(r=2.0), Target(r=2.0 + gap * r.range_res)])[0, 0]
    s = np.abs(range_fft(c, pad=8))[: 8 * r.n_samples // 2]
    pk = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > 0.5 * s.max()]
    res.append((gap, len(pk)))
    print(f"gap {gap:.1f} x dR -> {len(pk)} peak(s)")
if not o.no_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for gap in [2.0, 1.0, 0.5]:
        c = cube(r, [Target(r=2.0), Target(r=2.0 + gap * r.range_res)])[0, 0]
        s = np.abs(range_fft(c, pad=8)); x = np.arange(len(s)) * r.range_res / 8
        ax.plot(x, 20 * np.log10(s + 1e-12), label=f"gap {gap} x dR")
    ax.set_xlim(1.7, 2.4); ax.set_xlabel("range (m)"); ax.legend(); show(fig, "ch06_resolution.png")
