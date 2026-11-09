"""Chapter 20 - choosing 24, 60 or 77-81 GHz: the physics that follows from the band.

Run:  python examples/ch20_frequency_bands.py  [--no-plot]
Prints the numbers used in the article: wavelength, lambda/2 element spacing, range resolution from the usable
sweep, phase-cycle length, antenna gain of a fixed aperture, beamwidth of a fixed aperture, Doppler shift per m/s,
and the two-way oxygen loss at 60 GHz over typical indoor and outdoor ranges.
Regulatory band edges are quoted in the article with their sources; they are not computed here.
"""
import numpy as np
from _common import args, show

C = 299_792_458.0
BANDS = {   # label: (centre frequency for wavelength, usable sweep for a typical FMCW radar in that band)
    "24 GHz ISM": (24.125e9, 0.25e9),
    "60 GHz": (60.5e9, 4.0e9),          # 57-64 GHz; many radar chips sweep up to about 4 GHz of it
    "77-81 GHz": (78.5e9, 4.0e9),       # 77-81 GHz short-range band (5 GHz if 76-81 is usable)
}


def gain_dbi(f, area_m2, eff=0.6):
    lam = C / f
    return 10 * np.log10(4 * np.pi * area_m2 * eff / lam ** 2)


def beamwidth_deg(f, d_m):
    """-3 dB beamwidth of a uniformly illuminated aperture of length d: ~0.886 lambda/d (rad)."""
    return np.degrees(0.886 * C / f / d_m)


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    for k, (f, b) in BANDS.items():
        lam = C / f
        print(f"{k:11s}: lambda {lam*1e3:5.2f} mm, lambda/2 {lam*1e3/2:4.2f} mm, range res c/2B {C/(2*b)*100:5.1f} cm "
              f"(B {b/1e9:g} GHz), Doppler {2/lam:5.0f} Hz per m/s, "
              f"gain of 20x20 mm aperture {gain_dbi(f, 0.02*0.02):4.1f} dBi, beamwidth of 20 mm {beamwidth_deg(f, 0.02):4.1f} deg")
    print(f"76-81 full 5 GHz: {C/(2*5e9)*100:.1f} cm; 57-64 full 7 GHz: {C/(2*7e9)*100:.2f} cm")
    print(f"gain ratio 78.5 vs 24.125 for same aperture: {20*np.log10(78.5/24.125):.1f} dB; 60.5 vs 24.125: {20*np.log10(60.5/24.125):.1f} dB")
    for r in (5, 10, 30, 100):
        print(f"oxygen ~15 dB/km at 60 GHz: two-way loss at {r} m = {2*r*15/1000:.2f} dB")
    # aperture needed for a 10 deg beam
    for k, (f, b) in BANDS.items():
        print(f"{k}: aperture for a 10 deg beam {0.886*C/f/np.radians(10)*1000:.0f} mm")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        names = list(BANDS); lam = np.array([C / BANDS[k][0] for k in names])
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].bar(names, [C / (2 * BANDS[k][1]) * 100 for k in names]); ax[0].set_title("range resolution (cm)")
        ax[1].bar(names, [gain_dbi(BANDS[k][0], 4e-4) for k in names]); ax[1].set_title("gain of 20x20 mm (dBi)")
        ax[2].bar(names, [beamwidth_deg(BANDS[k][0], 0.02) for k in names]); ax[2].set_title("beam of 20 mm (deg)")
        fig.tight_layout(); show(fig, "ch20_bands.png")
