"""Chapter 4 - antenna gain and beamwidth from aperture size.

Run:  python examples/ch12_antenna_aperture.py  [--no-plot]
G = 4*pi*A*eta/lambda^2 for an aperture of area A and efficiency eta; -3 dB beamwidth of a uniformly illuminated
aperture of length D ~ 0.886 lambda/D; first sidelobe of a uniform aperture -13.3 dB; rough directivity from
beamwidths ~ 41253 / (thetaE * thetaH) in degrees. Textbook formulas, not any product.
"""
import numpy as np
from _common import args, show

C = 299_792_458.0


def gain_dbi(f, area_m2, eff=0.6):
    return 10 * np.log10(4 * np.pi * area_m2 * eff / (C / f) ** 2)


def beam_deg(f, d_m):
    return np.degrees(0.886 * C / f / d_m)


def kraus_dbi(th_e, th_h):
    return 10 * np.log10(41253 / (th_e * th_h))


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    for f in (60.5e9, 78.5e9):
        lam = C / f * 1e3
        print(f"{f/1e9:.1f} GHz: lambda {lam:.2f} mm")
        for d in (2.5, 5, 10, 20, 40):
            print(f"   {d:4.1f} mm square aperture: {gain_dbi(f, (d/1e3)**2):5.1f} dBi, beam {beam_deg(f, d/1e3):5.1f} deg")
        print(f"   aperture for a 10 deg beam: {0.886*lam/np.radians(10):.0f} mm")
    print(f"rough directivity of a +-60 deg (120 x 120 deg) beam: {kraus_dbi(120, 120):.1f} dBi")
    print(f"uniform aperture first sidelobe: {20*np.log10(abs(np.sinc(1.4303))):.1f} dB")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        d = np.linspace(2, 60, 300)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for f in (60.5e9, 78.5e9):
            ax[0].plot(d, gain_dbi(f, (d / 1e3) ** 2), label=f"{f/1e9:.1f} GHz")
            ax[1].plot(d, beam_deg(f, d / 1e3), label=f"{f/1e9:.1f} GHz")
        ax[0].set_ylabel("gain (dBi)"); ax[1].set_ylabel("-3 dB beamwidth (deg)"); ax[1].set_ylim(0, 80)
        for a in ax: a.set_xlabel("aperture side (mm)"); a.legend()
        fig.tight_layout(); show(fig, "ch04_gain_beam.png")
