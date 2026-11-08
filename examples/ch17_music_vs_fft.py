"""Chapter 17: two targets 8 deg apart — FFT/Bartlett vs Capon vs MUSIC on an 8-element half-wavelength array."""
import numpy as np
from _common import args, show
from radar_sim import angle_spectrum, capon, music

o = args(__doc__).parse_args()
pos = np.arange(8) * 0.5; rng = np.random.default_rng(1)
th = np.radians([0, 8]); A = np.exp(1j * 2 * np.pi * np.outer(pos, np.sin(th)))
S = (rng.standard_normal((2, 200)) + 1j * rng.standard_normal((2, 200))) / np.sqrt(2)
X = A @ S + 0.05 * (rng.standard_normal((8, 200)) + 1j * rng.standard_normal((8, 200)))
R = X @ X.conj().T / 200
ang = np.linspace(-30, 30, 1201)
def npk(p): p = p / p.max(); return sum(1 for i in range(1, len(p) - 1) if p[i] > p[i-1] and p[i] > p[i+1] and p[i] > 0.1)
bart = np.array([np.real(np.exp(-1j*2*np.pi*pos*np.sin(np.radians(a))) @ R @ np.exp(1j*2*np.pi*pos*np.sin(np.radians(a)))) for a in ang])
print("peaks found  Bartlett:", npk(bart), " Capon:", npk(capon(R, pos, ang)), " MUSIC:", npk(music(R, pos, ang, 2)))
