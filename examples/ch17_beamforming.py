"""Chapter 17 - beamforming and phased arrays (textbook level).

Run:  python examples/ch17_beamforming.py  [--no-plot]
Prints the numbers used in the article:
  1. progressive phase for steering, array gain, beamwidth vs N and scan angle
  2. grating-lobe condition d < lambda / (1 + |sin theta_max|)
  3. sidelobe level and beamwidth for uniform / Hamming / Taylor-like (Chebyshev) tapers
  4. beam squint of a phase-shifter array over a sweep vs true time delay
  5. phase-shifter quantisation (n-bit): pointing and sidelobe effect
  6. MVDR (Capon) adaptive beamformer: null depth on an interferer vs delay-and-sum
Textbook model; uniform linear array of isotropic elements unless stated. Not any product.
"""
import numpy as np
from _common import args, show
from numpy.linalg import inv

C = 299_792_458.0
FC = 60.5e9


def steer(pos_lam, th_deg):
    return np.exp(2j * np.pi * np.outer(np.sin(np.radians(np.atleast_1d(th_deg))), pos_lam))


def array_factor(w, pos_lam, th_deg, f_ratio=1.0):
    """Pattern |w^H a(theta)| normalised to 1 at its peak; f_ratio = f/f0 scales positions in wavelengths."""
    a = steer(pos_lam * f_ratio, th_deg)
    af = np.abs(a @ w.conj())
    return af / af.max()


def weights(n, d=0.5, th0=0.0, taper=None):
    pos = np.arange(n) * d
    t = np.ones(n) if taper is None else taper(n)
    return t * steer(pos, th0)[0], pos


def chebwin(n, at_db=30):
    """Dolph-Chebyshev taper (scipy): equal sidelobes at -at_db."""
    from scipy.signal.windows import chebwin as cw
    return cw(n, at_db)


def beamwidth_deg(pat, th):
    """-3 dB width of the main lobe around the peak."""
    i = np.argmax(pat); lvl = 1 / np.sqrt(2)
    lo = i
    while lo > 0 and pat[lo] > lvl: lo -= 1
    hi = i
    while hi < len(pat) - 1 and pat[hi] > lvl: hi += 1
    return th[hi] - th[lo]


def sidelobe_db(pat, th):
    i = np.argmax(pat)
    lo = i
    while lo > 0 and pat[lo - 1] < pat[lo]: lo -= 1
    hi = i
    while hi < len(pat) - 1 and pat[hi + 1] < pat[hi]: hi += 1
    rest = np.concatenate([pat[:lo], pat[hi + 1:]])
    return 20 * np.log10(rest.max())


def grating_spacing(th_max_deg):
    return 1 / (1 + abs(np.sin(np.radians(th_max_deg))))


def squint(th0_deg, f_ratio):
    """Phase-shifter array set for theta0 at f0: beam points where sin(theta) = sin(theta0)/f_ratio."""
    return np.degrees(np.arcsin(np.sin(np.radians(th0_deg)) / f_ratio))


def quantise(w, bits):
    step = 2 * np.pi / 2 ** bits
    ph = np.round(np.angle(w) / step) * step
    return np.abs(w) * np.exp(1j * ph)


def mvdr(R, a):
    Ri = inv(R)
    return Ri @ a / (a.conj() @ Ri @ a)


def mvdr_demo(n=8, th_sig=0.0, th_int=25.0, inr_db=30, snr_db=0, snaps=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.arange(n) * 0.5
    a_s, a_i = steer(pos, th_sig)[0], steer(pos, th_int)[0]
    s = 10 ** (snr_db / 20) * (rng.standard_normal(snaps) + 1j * rng.standard_normal(snaps)) / np.sqrt(2)
    it = 10 ** (inr_db / 20) * (rng.standard_normal(snaps) + 1j * rng.standard_normal(snaps)) / np.sqrt(2)
    nz = (rng.standard_normal((n, snaps)) + 1j * rng.standard_normal((n, snaps))) / np.sqrt(2)
    X = np.outer(a_s, s) + np.outer(a_i, it) + nz
    Xi = np.outer(a_i, it) + nz                        # interference + noise (training without the target)
    R = Xi @ Xi.conj().T / snaps
    w_das = a_s / n
    w_mv = mvdr(R, a_s)
    def sinr(w):
        ps = np.abs(w.conj() @ a_s) ** 2 * 10 ** (snr_db / 10)
        pin = np.real(w.conj() @ (10 ** (inr_db / 10) * np.outer(a_i, a_i.conj()) + np.eye(n)) @ w)
        return 10 * np.log10(ps / pin)
    return pos, w_das, w_mv, sinr(w_das), sinr(w_mv)


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    lam = C / FC
    th = np.linspace(-90, 90, 36001)
    print(f"lambda {lam*1e3:.3f} mm")
    for th0 in (0, 30, 45):
        print(f"steer to {th0} deg: phase step per element (d=lambda/2) {180*np.sin(np.radians(th0)):.1f} deg")
    for n in (4, 8, 16, 32):
        for th0 in (0, 45):
            w, pos = weights(n, th0=th0)
            print(f"N={n} scan {th0}: -3 dB beamwidth {beamwidth_deg(array_factor(w, pos, th), th):.2f} deg, "
                  f"array gain {10*np.log10(n):.1f} dB (rx) / {20*np.log10(n):.1f} dB EIRP (tx, N PAs)")
    for tm in (30, 45, 60, 90):
        print(f"no grating lobe up to +/-{tm} deg scan: d < {grating_spacing(tm):.3f} lambda")
    w, pos = weights(8, d=1.0, th0=40)
    pat = array_factor(w, pos, th); pk = th[pat > 0.99]
    print("d = lambda, scan 40: beams at", np.unique(np.round(pk)))
    for name, tp in (("uniform", None), ("hamming", np.hamming), ("cheb30", lambda n: chebwin(n, 30))):
        w, pos = weights(16, taper=tp)
        p = array_factor(w, pos, th)
        print(f"16 elements {name}: sidelobe {sidelobe_db(p, th):.1f} dB, beamwidth {beamwidth_deg(p, th):.2f} deg")
    for f in (58e9, 62e9, 57e9, 64e9):
        print(f"phase-shifter array set to 45 deg at 60.5 GHz, at {f/1e9} GHz points to {squint(45, f/FC):.2f} deg")
    for f in (76e9, 81e9):
        print(f"77-81: set 45 deg at 78.5, at {f/1e9} GHz -> {squint(45, f/78.5e9):.2f} deg")
    for bits in (2, 3, 4, 6):
        w, pos = weights(16, th0=17)
        p = array_factor(quantise(w, bits), pos, th)
        print(f"{bits}-bit phase shifters, 16 el, steer 17 deg: peak {th[np.argmax(p)]:.2f} deg, worst sidelobe {sidelobe_db(p, th):.1f} dB")
    pos, wd, wm, s1, s2 = mvdr_demo()
    pd_ = 20 * np.log10(array_factor(wd, pos, np.array([25.0, 0.0])) + 1e-12)
    pm_ = 20 * np.log10(np.abs(steer(pos, [25.0, 0.0]) @ wm.conj()) / np.abs(steer(pos, [0.0])[0] @ wm.conj()))
    print(f"MVDR demo: gain toward 25 deg interferer: delay-and-sum {pd_[0]:.1f} dB, MVDR {pm_[0]:.1f} dB; "
          f"output SINR {s1:.1f} dB vs {s2:.1f} dB")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for th0 in (0, 30, 60):
            w, pos = weights(16, th0=th0)
            ax.plot(th, 20 * np.log10(array_factor(w, pos, th) + 1e-6), label=f"16 elements, steered {th0} deg")
        w, pos = weights(16, taper=lambda n: chebwin(n, 30))
        ax.plot(th, 20 * np.log10(array_factor(w, pos, th) + 1e-6), "k--", label="Chebyshev 30 dB taper, 0 deg")
        ax.set_ylim(-50, 2); ax.set_xlabel("angle (deg)"); ax.set_ylabel("pattern (dB)"); ax.legend(); fig.tight_layout()
        show(fig, "ch17_beam_patterns.png")
