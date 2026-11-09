"""Chapter 1 - wavelength and the sizes that follow from it.

Run:  python examples/ch01_wavelength.py  [--no-plot]
Prints wavelength, lambda/2 element spacing, one phase turn of radial motion (lambda/2) and the motion per degree
of phase for common radar bands, plus the wavelength inside a dielectric (lambda / sqrt(er)).
"""
import numpy as np
from _common import args, show

C = 299_792_458.0
BANDS = {"5.8 GHz": 5.8e9, "24 GHz": 24.125e9, "60 GHz": 60.5e9, "77 GHz": 77e9, "79 GHz": 78.5e9}

if __name__ == "__main__":
    o = args(__doc__).parse_args()
    for k, f in BANDS.items():
        lam = C / f
        print(f"{k:8s}: lambda {lam*1e3:6.2f} mm, lambda/2 {lam*1e3/2:5.2f} mm, "
              f"phase turn per {lam/2*1e3:5.2f} mm of radial motion, {lam/720*1e6:5.2f} um per degree")
    for er in (2.1, 2.8, 4.4):
        print(f"in a dielectric with er {er}: 60.5 GHz wavelength {C/60.5e9/np.sqrt(er)*1e3:.2f} mm, "
              f"78.5 GHz {C/78.5e9/np.sqrt(er)*1e3:.2f} mm")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        x = np.linspace(0, 50, 2000)
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (k, f) in enumerate(list(BANDS.items())[1:]):
            lam = C / f * 1e3
            ax.plot(x, 2 * i + 0.4 * np.sin(2 * np.pi * x / lam), label=f"{k}: {lam:.2f} mm")
        ax.set_xlabel("mm (to scale)"); ax.set_yticks([]); ax.legend(loc="upper right"); fig.tight_layout()
        show(fig, "ch01_wavelengths.png")
