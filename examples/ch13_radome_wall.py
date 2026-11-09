"""Chapter 13 - radome walls: half-wave thickness, reflection, and the angle error from an uneven cover.

Run:  python examples/ch13_radome_wall.py  [--no-plot]
1. normal-incidence reflection of a lossless dielectric slab versus thickness (zero at k * lambda / (2 sqrt(er)))
2. angle bias when the wall is dt thicker over one of two lambda/2-spaced receivers: extra phase
   2*pi*dt*(sqrt(er)-1)/lambda, read by the radar as a tilted wavefront.
Single-layer, lossless, normal-incidence model; not any product.
"""
import numpy as np
from _common import args, show

C = 299_792_458.0


def slab_gamma(t_m, f, er):
    n = np.sqrt(er); r = (1 - n) / (1 + n)
    e = np.exp(-2j * 2 * np.pi * f / C * n * t_m)
    return r * (1 - e) / (1 - r * r * e)


def half_wave_mm(f, er, k=1):
    return k * C / f / (2 * np.sqrt(er)) * 1e3


def angle_bias_deg(dt_m, f, er, d_lambda=0.5, theta0=0.0):
    dphi = 2 * np.pi * dt_m * (np.sqrt(er) - 1) / (C / f)
    s = np.sin(np.radians(theta0)) + dphi / (2 * np.pi * d_lambda)
    return np.degrees(np.arcsin(s)) - theta0, np.degrees(dphi)


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    for f in (24.125e9, 60.5e9, 78.5e9):
        print(f"PC (er 2.8) at {f/1e9:.3f} GHz: half-wave wall {half_wave_mm(f, 2.8):.2f} mm, "
              f"worst reflection {20*np.log10(abs(slab_gamma(half_wave_mm(f, 2.8)/2e3, f, 2.8))):.1f} dB at quarter-wave")
    t = 1.14e-3
    for err in (0.05e-3, 0.1e-3):
        print(f"78.5 GHz wall {t*1e3:.2f} mm off by {err*1e3:.2f} mm: reflection "
              f"{20*np.log10(abs(slab_gamma(t + err, 78.5e9, 2.8))):.1f} dB")
    for dt in (0.05e-3, 0.1e-3, 0.2e-3):
        for f in (60.5e9, 78.5e9):
            b, p = angle_bias_deg(dt, f, 2.8)
            print(f"dt {dt*1e3:.2f} mm at {f/1e9:.1f} GHz: {p:4.1f} deg phase -> {b:4.2f} deg angle error at boresight")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        tt = np.linspace(0, 3e-3, 1000)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for f in (60.5e9, 78.5e9):
            ax[0].plot(tt * 1e3, 20 * np.log10(np.abs(slab_gamma(tt, f, 2.8)) + 1e-9), label=f"{f/1e9:.1f} GHz")
            dt = np.linspace(0, 0.3e-3, 100)
            ax[1].plot(dt * 1e3, angle_bias_deg(dt, f, 2.8)[0], label=f"{f/1e9:.1f} GHz")
        ax[0].set_ylim(-40, 0); ax[0].set_xlabel("PC wall thickness (mm)"); ax[0].set_ylabel("reflection (dB)")
        ax[1].set_xlabel("extra thickness over one RX (mm)"); ax[1].set_ylabel("angle error (deg)")
        for a in ax: a.legend()
        fig.tight_layout(); show(fig, "ch13_radome.png")
