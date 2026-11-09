"""Chapter 2: echo phase 4*pi*R/lambda, I/Q, and displacement from unwrapped phase.
Moves a target by a fraction of a wavelength and recovers the motion from the range-bin phase."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, range_fft, C

a = args(__doc__); a.add_argument("--amp-mm", type=float, default=3.0, help="peak displacement (mm)"); o = a.parse_args()
r = make("1T2R", n_chirps=1)
lam = r.lam
t = np.linspace(0, 2, 200)                                   # 2 s, 100 frames/s
disp = o.amp_mm * 1e-3 * np.sin(2 * np.pi * 0.5 * t)         # slow sinusoidal motion
iq = []
for d in disp:
    rp = range_fft(cube(r, [Target(r=1.0 + d)], motion=False)[0, 0])
    k = int(round(1.0 / r.range_res))
    iq.append(rp[k])
iq = np.array(iq)
phi = np.unwrap(np.angle(iq))
# The bin phase follows the carrier at the centre of the sampled sweep (the beat term 2*pi*fb*t adds
# 4*pi*R*S*t/c across the ramp). The model's fc is that centre frequency, so lambda = c/fc converts directly.
est = (phi - phi[0]) * lam / (4 * np.pi) + disp[0]
print(f"wavelength {lam*1e3:.2f} mm, one phase turn per {lam/2*1e3:.2f} mm, {lam/720*1e6:.1f} um per degree")
print(f"peak phase swing {np.degrees(phi.max()-phi.min())/2:.0f} deg, RMS error of recovered displacement (noise-free model) {np.sqrt(np.mean((est-disp)**2))*1e6:.2f} um")
if not o.no_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(iq.real, iq.imag); ax[0].set_aspect("equal"); ax[0].set_xlabel("I"); ax[0].set_ylabel("Q"); ax[0].set_title("range-bin I/Q")
    ax[1].plot(t, disp * 1e3, label="true"); ax[1].plot(t, est * 1e3, "--", label="from phase"); ax[1].legend(); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("mm")
    fig.tight_layout(); show(fig, "ch02_phase.png")
