"""A real-time-style FMCW processing chain: range-Doppler -> 2-D CFAR -> angle at detections -> tracks.

Textbook building blocks, written for reading rather than speed:
  ca_cfar_2d            cell-averaging CFAR on a range-Doppler power map (wraps in Doppler, not in range)
  peaks_2d              keep only local maxima among CFAR hits
  angle_fft_all         reference: angle FFT over the virtual channels for EVERY range-Doppler cell (a "3D FFT")
  angle_dft_at          DFT over the virtual channels only at detected cells, on a coarse angle grid
  angle_split_phase     coarse angle from a small subarray, refined by the phase between the two array halves
                        (split-aperture phase comparison, as in phase-comparison monopulse)
  tdm_phase_comp        undo the Doppler phase step between TDM-MIMO transmit slots before angle processing
  beam_monopulse        analog / AESA style: the array is phase-steered to one beam position and only two
                        half-array sums reach the ADC; fine angle = beam position + phase-comparison offset
  op_counts             complex multiply counts for the three digital angle strategies
  Tracker               multi-target constant-velocity Kalman tracker with gating and M-of-N confirmation
All angles in degrees; virtual positions in wavelengths along one line (azimuth row).
"""
import numpy as np


# ------------------------------------------------------------------ detection
def _box_sum(p, kd, kr):
    """Sum of p over a (2kd+1) x (2kr+1) box; wraps in Doppler (axis 0), zero-padded in range (axis 1).
    Also returns how many real cells each box covered (fewer near the range edges)."""
    q = np.pad(p, ((kd, kd), (0, 0)), mode="wrap")
    q = np.pad(q, ((0, 0), (kr, kr)))
    ones = np.pad(np.ones_like(p), ((kd, kd), (0, 0)), mode="wrap")
    ones = np.pad(ones, ((0, 0), (kr, kr)))
    def box(a):
        c = np.cumsum(np.cumsum(a, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        h, w = 2 * kd + 1, 2 * kr + 1
        return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    return box(q), box(ones)


def ca_cfar_2d(p, guard=(2, 2), train=(4, 8), pfa=1e-6):
    """2-D CA-CFAR on a power map p[doppler, range]. guard/train = cells on EACH side (doppler, range).
    Threshold = alpha * mean(training cells), alpha set for the nominal number of training cells.
    Returns (hits, threshold)."""
    gd, gr = guard; td, tr = train
    outer, n_out = _box_sum(p, gd + td, gr + tr)
    inner, n_in = _box_sum(p, gd, gr)
    n = n_out - n_in
    n_nom = (2 * (gd + td) + 1) * (2 * (gr + tr) + 1) - (2 * gd + 1) * (2 * gr + 1)
    alpha = n_nom * (pfa ** (-1 / n_nom) - 1)
    th = alpha * (outer - inner) / np.maximum(n, 1)
    return p > th, th


def peaks_2d(p, hits):
    """Keep CFAR hits that are the maximum of their 3x3 neighbourhood (Doppler wraps)."""
    q = np.pad(p, ((1, 1), (0, 0)), mode="wrap")
    q = np.pad(q, ((0, 0), (1, 1)), constant_values=-np.inf)
    nb = np.max([q[1 + i:q.shape[0] - 1 + i, 1 + j:q.shape[1] - 1 + j]
                 for i in (-1, 0, 1) for j in (-1, 0, 1)], axis=0)
    return hits & (p >= nb)


# ------------------------------------------------------------------ angle
def steering(pos, ang_deg):
    return np.exp(2j * np.pi * np.outer(np.sin(np.radians(np.atleast_1d(ang_deg))), pos))


def angle_fft_all(rd, n_fft=64):
    """Reference "3D FFT": FFT across the virtual-channel axis of the whole cube rd[doppler, channel, range]
    (uniform lambda/2 row assumed). Returns |spectrum|[doppler, angle_bin, range] and the bin angles."""
    s = np.fft.fftshift(np.fft.fft(rd, n_fft, axis=1), axes=1)
    u = (np.arange(n_fft) - n_fft // 2) / n_fft * 2          # sin(theta) for d = lambda/2
    return np.abs(s), np.degrees(np.arcsin(np.clip(u, -1, 1)))


def angle_dft_at(snap, pos, grid_deg):
    """Bartlett/DFT scan of one snapshot over an angle grid. Returns the grid angle with the largest response."""
    p = np.abs(steering(pos, grid_deg).conj() @ snap)
    return grid_deg[int(np.argmax(p))], p


def angle_split_phase(snap, pos, coarse_grid_deg, sub=4):
    """1. coarse angle: DFT over the first `sub` channels on a coarse grid (cheap, wide beam)
       2. fine angle: form two half-array beams steered to the coarse angle, take the phase between them.
          Their phase centres are B = (N/2)*d apart, so dphi = 2*pi*B*(sin th - sin th0); solve for th.
          The coarse step must keep the residual inside the unambiguous +-lambda/(2B) window."""
    n = len(pos)
    th0, _ = angle_dft_at(snap[:sub], pos[:sub], coarse_grid_deg)
    a0 = steering(pos, th0)[0]
    h = n // 2
    left = a0[:h].conj() @ snap[:h]
    right = a0[h:].conj() @ snap[h:]
    B = pos[h:].mean() - pos[:h].mean()
    dphi = np.angle(right * np.conj(left))
    s = np.sin(np.radians(th0)) + dphi / (2 * np.pi * B)
    return float(np.degrees(np.arcsin(np.clip(s, -1, 1)))), th0


def tdm_phase_comp(snap, d, n_chirps, n_tx, n_rx):
    """TDM-MIMO: transmitter k fires k chirp slots after transmitter 0, so a mover adds a phase
    2*pi*fd*k*Tc to its channels. With the Doppler FFT taken per transmitter (n_chirps loops), the shifted
    bin index d gives that phase as 2*pi*(d - n_chirps/2)/(n_chirps*n_tx) per slot (valid inside the
    unambiguous Doppler span; outside it, see chapter 16)."""
    k = np.repeat(np.arange(n_tx), n_rx)
    return snap * np.exp(-2j * np.pi * (d - n_chirps // 2) * k / (n_chirps * n_tx))


def beam_monopulse(elem, pos, beam_grid_deg):
    """Analog beam scan + phase comparison. elem = signal at each element (before any combining).
    For every beam position the hardware would form two half-array sums with its phase shifters;
    the beam with the largest |left + right| is kept and the angle inside it comes from the phase between
    the halves. Returns (angle, chosen beam). On real hardware the beams are visited one dwell at a time."""
    h = len(pos) // 2
    a = steering(pos, beam_grid_deg)
    left = a[:, :h].conj() @ elem[:h]
    right = a[:, h:].conj() @ elem[h:]
    i = int(np.argmax(np.abs(left + right)))
    B = pos[h:].mean() - pos[:h].mean()
    s = np.sin(np.radians(beam_grid_deg[i])) + np.angle(right[i] * np.conj(left[i])) / (2 * np.pi * B)
    return float(np.degrees(np.arcsin(np.clip(s, -1, 1)))), float(beam_grid_deg[i])


def op_counts(n_dopp, n_range, n_virt, n_det, n_fft=64, coarse_bins=16, sub=4):
    """Complex multiplies for the angle step only.
    full:  an n_fft-point FFT over channels for every range-Doppler cell (~ n_fft/2 * log2 n_fft each)
    dft:   n_virt x n_ang direct DFT, only at detections (n_ang = n_fft)
    split: sub x coarse_bins DFT + two n_virt/2 beams at detections"""
    full = n_dopp * n_range * (n_fft / 2) * np.log2(n_fft)
    dft = n_det * n_virt * n_fft
    split = n_det * (sub * coarse_bins + n_virt)
    return {"full_3d_fft": full, "dft_at_detections": dft, "split_phase_at_detections": split}


# ------------------------------------------------------------------ tracking
class KalmanCV:
    """Constant-velocity Kalman filter on a 2-D position measurement. State [x, y, vx, vy]."""

    def __init__(self, z0, dt, q=2.0, r=0.15):
        self.x = np.array([z0[0], z0[1], 0.0, 0.0])
        self.P = np.diag([r * r, r * r, 4.0, 4.0])
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = dt
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.Q = q * q * g @ g.T
        self.H = np.eye(2, 4)
        self.R = r * r * np.eye(2)

    def predict(self):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q

    def gate(self, z):
        """Mahalanobis distance squared of measurement z (chi-square, 2 dof)."""
        S = self.H @ self.P @ self.H.T + self.R
        v = z - self.H @ self.x
        return float(v @ np.linalg.solve(S, v))

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


class Tracker:
    """Global-nearest-neighbour multi-target tracker.
    New tracks start tentative from unassociated detections, are confirmed after M hits in the last N frames,
    and are deleted after `drop` consecutive misses. Gate = chi-square 2 dof (9.21 ~ 99 %)."""

    def __init__(self, dt, confirm=(3, 5), drop=4, gate=9.21, q=2.0, r=0.15):
        self.dt, self.m, self.n, self.drop, self.g, self.q, self.r = dt, *confirm, drop, gate, q, r
        self.tracks, self.next_id = [], 1

    def step(self, dets):
        dets = [np.asarray(d, float) for d in dets]
        for t in self.tracks:
            t["kf"].predict()
        pairs = sorted((t["kf"].gate(d), i, j) for i, t in enumerate(self.tracks) for j, d in enumerate(dets))
        used_t, used_d = set(), set()
        for d2, i, j in pairs:
            if d2 > self.g or i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            self.tracks[i]["kf"].update(dets[j])
        for i, t in enumerate(self.tracks):
            hit = i in used_t
            t["hist"] = (t["hist"] + [hit])[-self.n:]
            t["miss"] = 0 if hit else t["miss"] + 1
            if not t["confirmed"] and sum(t["hist"]) >= self.m:
                t["confirmed"] = True
        self.tracks = [t for t in self.tracks if t["miss"] < self.drop and
                       (t["confirmed"] or len(t["hist"]) < self.n or sum(t["hist"]) >= self.m - (self.n - len(t["hist"])) or t["miss"] < 2)]
        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks.append({"id": self.next_id, "kf": KalmanCV(d, self.dt, self.q, self.r),
                                    "hist": [True], "miss": 0, "confirmed": False})
                self.next_id += 1
        return [(t["id"], t["kf"].x.copy()) for t in self.tracks if t["confirmed"]]


# ------------------------------------------------------------------ single-cell Doppler DFT
def doppler_dft_at(rc, r, d, win=True):
    """Value of the range-Doppler map at one cell, computed from range-FFT output rc[chirp, ..., range] alone:
    z = sum_m w[m] X[m, r] exp(-j 2 pi m (d - M/2) / M), with d the fftshift-ed Doppler index.
    Identical to doppler_fft(rc)[d, ..., r] but costs M multiplies per channel instead of a whole FFT.
    Lets a second receive path skip its own Doppler FFT (and corner-turn memory) and still give a
    coherent value at every detection."""
    m = rc.shape[0]
    w = np.hanning(m) if win else np.ones(m)
    k = np.exp(-2j * np.pi * np.arange(m) * (d - m // 2) / m)
    return np.tensordot(w * k, rc[..., r], axes=(0, 0)) / w.sum()


def to_xy(r, az_deg):
    a = np.radians(az_deg)
    return np.array([r * np.sin(a), r * np.cos(a)])
