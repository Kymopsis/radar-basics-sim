"""Minimal FMCW MIMO radar simulator for the Radar Basics series (textbook model, not any product).

Signal model (dechirped, complex baseband, ideal point targets, TDM-MIMO):
  s[chirp m, virtual channel k, sample n] =
     a * exp(j*2*pi*( fb*n/fs + fd*m*Tc_eff + (2/lam)*R0 ... ))
Beat frequency fb = 2*S*R/c, Doppler fd = 2*v/lam, phase 4*pi*R/lam, angle term
2*pi*(d_k . u)/lam where d_k is the virtual element position and u the unit direction.
"""
from dataclasses import dataclass, field
import numpy as np

C = 299_792_458.0


@dataclass
class Radar:
    fc: float = 60.5e9          # centre frequency (Hz)
    bw: float = 4e9             # sweep bandwidth (Hz)
    tc: float = 40e-6           # ramp time (s)
    fs: float = 6.4e6           # ADC rate (Hz)
    n_samples: int = 256
    n_chirps: int = 64
    t_idle: float = 10e-6       # idle between ramps (s)
    tx: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0]]))   # (n_tx, 2) in units of lambda: (x, z)
    rx: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0], [0.5, 0.0]]))
    noise_dbfs: float = -90.0   # noise power per sample relative to a unit-amplitude echo, dB

    @property
    def lam(self): return C / self.fc
    @property
    def slope(self): return self.bw / self.tc
    @property
    def t_chirp(self): return self.tc + self.t_idle
    @property
    def t_frame_chirp(self):    # time between chirps of the SAME tx (TDM)
        return self.t_chirp * len(self.tx)
    @property
    def range_res(self): return C / (2 * self.bw * self.n_samples / (self.fs * self.tc))
    @property
    def r_max(self): return self.fs * C / (2 * self.slope)        # complex ADC
    @property
    def v_max(self): return self.lam / (4 * self.t_frame_chirp)
    @property
    def v_res(self): return self.lam / (2 * self.n_chirps * self.t_frame_chirp)

    def virtual(self):
        """Virtual array = every tx position + every rx position (in lambda), ordered tx-major."""
        return np.array([t + r for t in self.tx for r in self.rx])

    def summary(self):
        return (f"fc {self.fc/1e9:.1f} GHz, B {self.bw/1e9:.1f} GHz, ΔR {self.range_res*100:.2f} cm, "
                f"Rmax {self.r_max:.1f} m, vmax ±{self.v_max:.2f} m/s, Δv {self.v_res*100:.1f} cm/s, "
                f"{len(self.tx)}T{len(self.rx)}R")


@dataclass
class Target:
    r: float                 # range (m)
    v: float = 0.0           # radial velocity (m/s, + = receding)
    az: float = 0.0          # azimuth (deg)
    el: float = 0.0          # elevation (deg)
    amp: float = 1.0         # echo amplitude (linear, relative)


def cube(radar: Radar, targets, seed=0, motion=True):
    """Return complex cube [n_chirps, n_virtual, n_samples]."""
    rng = np.random.default_rng(seed)
    n = np.arange(radar.n_samples) / radar.fs
    virt = radar.virtual()
    ntx, nrx = len(radar.tx), len(radar.rx)
    out = np.zeros((radar.n_chirps, len(virt), radar.n_samples), complex)
    for t in targets:
        az, el = np.radians(t.az), np.radians(t.el)
        ux, uz = np.sin(az) * np.cos(el), np.sin(el)
        for m in range(radar.n_chirps):
            for it in range(ntx):
                t0 = m * radar.t_frame_chirp + it * radar.t_chirp
                r = t.r + (t.v * t0 if motion else 0.0)
                fb = 2 * radar.slope * r / C
                ph = 4 * np.pi * r / radar.lam
                for ir in range(nrx):
                    k = it * nrx + ir
                    d = virt[k]
                    sp = 2 * np.pi * (d[0] * ux + d[1] * uz)
                    out[m, k] += t.amp * np.exp(1j * (2 * np.pi * fb * n + ph + sp))
    sigma = 10 ** (radar.noise_dbfs / 20) / np.sqrt(2)
    out += sigma * (rng.standard_normal(out.shape) + 1j * rng.standard_normal(out.shape))
    return out


def db(x, floor=-120):
    return np.maximum(20 * np.log10(np.abs(x) + 1e-15), floor)


def range_fft(c, win=True, pad=1):
    n = c.shape[-1]
    w = np.hanning(n) if win else np.ones(n)
    return np.fft.fft(c * w, n * pad, axis=-1) / w.sum()


def doppler_fft(rc, win=True):
    n = rc.shape[0]
    w = (np.hanning(n) if win else np.ones(n))[:, None, None]
    return np.fft.fftshift(np.fft.fft(rc * w, axis=0), axes=0) / w.sum()


def rd_map(radar, c, channel=None):
    """Range-Doppler magnitude (non-coherent sum over channels unless channel given).
    Returns (map[doppler, range], ranges, velocities)."""
    rd = doppler_fft(range_fft(c))
    m = np.abs(rd[:, channel, :]) if channel is not None else np.sqrt((np.abs(rd) ** 2).mean(axis=1))
    nr = c.shape[-1]
    ranges = np.arange(nr) * radar.range_res
    vel = (np.arange(radar.n_chirps) - radar.n_chirps // 2) * radar.v_res
    return m, ranges, vel, rd


def angle_spectrum(snapshot, positions_lambda, angles_deg):
    """Bartlett (FFT-equivalent) spectrum along x for a 1-D set of element positions (lambda)."""
    a = np.exp(1j * 2 * np.pi * np.outer(np.sin(np.radians(angles_deg)), positions_lambda))
    return np.abs(a.conj() @ snapshot) / len(positions_lambda)


def music(R, positions_lambda, angles_deg, n_src):
    w, v = np.linalg.eigh(R)
    en = v[:, : R.shape[0] - n_src]
    a = np.exp(1j * 2 * np.pi * np.outer(np.sin(np.radians(angles_deg)), positions_lambda))
    p = 1 / np.sum(np.abs(a.conj() @ en) ** 2, axis=1)
    return p / p.max()


def capon(R, positions_lambda, angles_deg, load=1e-3):
    Ri = np.linalg.inv(R + load * np.trace(R).real / len(R) * np.eye(len(R)))
    a = np.exp(1j * 2 * np.pi * np.outer(np.sin(np.radians(angles_deg)), positions_lambda))
    p = 1 / np.real(np.einsum("ij,jk,ik->i", a.conj(), Ri, a))
    return p / p.max()


def ca_cfar_1d(p, guard=2, train=8, pfa=1e-4):
    n = len(p); out = np.zeros(n, bool); th = np.full(n, np.nan)
    N = 2 * train
    alpha = N * (pfa ** (-1 / N) - 1)
    for i in range(n):
        idx = [j % n for j in range(i - guard - train, i - guard)] + \
              [j % n for j in range(i + guard + 1, i + guard + train + 1)]
        th[i] = alpha * np.mean(p[idx])
        out[i] = p[i] > th[i]
    return out, th


# ---- generic textbook layouts (lambda units; x = azimuth, z = elevation) -----------------------
LAYOUTS = {
    "1T2R": dict(tx=[[0, 0]], rx=[[0, 0], [0.5, 0]]),
    "2T2R": dict(tx=[[0, 0], [1.0, 0]], rx=[[0, 0], [0.5, 0]]),
    "1T4R": dict(tx=[[0, 0]], rx=[[0, 0], [0.5, 0], [1.0, 0], [1.5, 0]]),
    "2T4R": dict(tx=[[0, 0], [2.0, 0]], rx=[[0, 0], [0.5, 0], [1.0, 0], [1.5, 0]]),
    # common textbook 3T4R arrangement: two TX in azimuth 2λ apart, a third raised λ/2 for elevation
    "3T4R": dict(tx=[[0, 0], [1.0, 0.5], [2.0, 0]], rx=[[0, 0], [0.5, 0], [1.0, 0], [1.5, 0]]),
}


def make(layout="1T2R", **kw):
    L = LAYOUTS[layout]
    return Radar(tx=np.array(L["tx"], float), rx=np.array(L["rx"], float), **kw)
