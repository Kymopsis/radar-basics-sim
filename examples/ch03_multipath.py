"""Chapter 3: two-ray multipath. Echo power vs distance for a radar and target above a reflecting floor,
and the spacing of the fading nulls. Change --hr, --ht, --gamma to try your own installation."""
import numpy as np
from _common import args, show
from radar_sim import C

a = args(__doc__)
a.add_argument("--fc", type=float, default=60.5e9); a.add_argument("--hr", type=float, default=1.0, help="radar height (m)")
a.add_argument("--ht", type=float, default=1.0, help="target height (m)"); a.add_argument("--gamma", type=float, default=-0.7)
o = a.parse_args()
lam = C / o.fc
d = np.linspace(1, 10, 20000)
delta = np.hypot(d, o.hr + o.ht) - np.hypot(d, o.hr - o.ht)
F = np.abs(1 + o.gamma * np.exp(-1j * 2 * np.pi * delta / lam))
gain_db = 40 * np.log10(F)                                   # two-way: power ~ F^4
print(f"echo vs free space: max {gain_db.max():+.1f} dB, min {gain_db.min():+.1f} dB")
for x in (2, 5, 10):
    print(f"null spacing near {x} m: {lam*x**2/(2*o.hr*o.ht)*100:.1f} cm")
if not o.no_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(d, gain_db, lw=0.8); ax.set_xlabel("distance (m)"); ax.set_ylabel("dB vs free space")
    fig.tight_layout(); show(fig, "ch03_two_ray.png")
