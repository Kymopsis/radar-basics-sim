"""Chapter 10: micro-Doppler spectrogram of a swinging scatterer and breathing rate from the echo phase.
Experimental illustration only; not a medical device."""
import numpy as np
from _common import args, show
from radar_sim import C

a = args(__doc__); a.add_argument("--fc", type=float, default=60.5e9); o = a.parse_args()
lam = C / o.fc; rng = np.random.default_rng(0)

# 1) micro-Doppler: torso at 1 m/s plus a limb swinging +-0.3 m at 0.9 Hz, slow-time rate 4 kHz
fs = 4000.0; t = np.arange(0, 3, 1 / fs)
torso = 3 + 1.0 * t; limb = torso + 0.3 * np.sin(2 * np.pi * 0.9 * t)
x = np.exp(1j * 4 * np.pi * torso / lam) + 0.3 * np.exp(1j * 4 * np.pi * limb / lam)
nwin, hop = 512, 32; w = np.hanning(nwin)
S = np.array([np.abs(np.fft.fftshift(np.fft.fft(x[s:s + nwin] * w))) for s in range(0, len(x) - nwin, hop)]).T
vel = (np.arange(nwin) - nwin // 2) / nwin * fs * lam / 2
limb_v = vel[np.argmax(S * (np.abs(vel - 1.0) > 0.3)[:, None], axis=0)]
print(f"micro-Doppler: torso 1.0 m/s, limb velocity spans {limb_v.min():+.1f} to {limb_v.max():+.1f} m/s")

# 2) breathing: 4 mm at 15/min plus 0.3 mm heartbeat at 72/min, frames at 20 Hz, static clutter in the same bin
ff = 20.0; tt = np.arange(0, 60, 1 / ff)
chest = 0.004 * np.sin(2 * np.pi * 0.25 * tt) + 0.0003 * np.sin(2 * np.pi * 1.2 * tt)
y = np.exp(1j * 4 * np.pi * (0.8 + chest) / lam) + 0.6 + 0.02 * (rng.standard_normal(tt.size) + 1j * rng.standard_normal(tt.size))
y = y - y.mean()                                             # remove static clutter
disp = np.unwrap(np.angle(y)) * lam / (4 * np.pi)
sp = np.abs(np.fft.rfft((disp - disp.mean()) * np.hanning(tt.size))); f = np.fft.rfftfreq(tt.size, 1 / ff)
br = f[np.argmax(sp * (f < 0.6))] * 60; hr = f[np.argmax(sp * ((f > 0.8) & (f < 2.0)))] * 60
print(f"breathing {br:.0f}/min, heartbeat band peak {hr:.0f}/min (simulated, idealised)")
if not o.no_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(20 * np.log10(S / S.max() + 1e-6), origin="lower", aspect="auto", vmin=-50, extent=[0, 3, vel[0], vel[-1]]); ax[0].set_ylim(-1.5, 3.5)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("velocity (m/s)"); ax[0].set_title("micro-Doppler")
    ax[1].plot(tt, disp * 1e3); ax[1].set_xlim(0, 20); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("chest (mm)"); ax[1].set_title("phase displacement")
    fig.tight_layout(); show(fig, "ch10_micro_doppler.png")
