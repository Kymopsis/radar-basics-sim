"""Chapter 16 - detection (thresholds, CFAR, clutter) and tracking (clustering, Kalman filter).

Run:  python examples/ch14_cfar_tracking.py  [--no-plot]
Prints the numbers used in the article:
  1. false-alarm probability of a fixed threshold in exponential (square-law) noise
  2. CA-CFAR scale factor and CFAR loss (Swerling 1 closed forms) + Monte Carlo check
  3. SNR needed for Pd 0.9 / Pfa 1e-6 (Albersheim, non-fluctuating target)
  4. false alarms per second for a given range-Doppler map size and frame rate
  5. a constant-velocity Kalman tracker on simulated detections with clutter false alarms
Textbook model, not any product. Uses radar_sim where it fits.
"""
import numpy as np
from _common import args, show
from radar_sim.fmcw import ca_cfar_1d


def pfa_fixed(t_over_noise):
    """Square-law detector, complex Gaussian noise: P(|n|^2 > T) = exp(-T/sigma^2)."""
    return np.exp(-t_over_noise)


def ca_alpha(n_train, pfa):
    return n_train * (pfa ** (-1 / n_train) - 1)


def pd_swerling1_known(snr, pfa):
    """Known noise level, Swerling-1 target: Pd = Pfa^(1/(1+SNR))."""
    return pfa ** (1 / (1 + snr))


def pd_swerling1_ca(snr, pfa, n_train):
    """CA-CFAR, Swerling-1: Pd = (1 + alpha/(N(1+SNR)))^-N."""
    a = ca_alpha(n_train, pfa)
    return (1 + a / (n_train * (1 + snr))) ** (-n_train)


def snr_needed(pd_fn, pd=0.9, lo=-10, hi=60):
    for _ in range(80):
        mid = (lo + hi) / 2
        if pd_fn(10 ** (mid / 10)) < pd: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def albersheim(pd, pfa, n=1):
    A = np.log(0.62 / pfa); B = np.log(pd / (1 - pd))
    return -5 * np.log10(n) + (6.2 + 4.54 / np.sqrt(n + 0.44)) * np.log10(A + 0.12 * A * B + 1.7 * B)


def mc_ca_pfa(n_train, pfa, trials=400000, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.exponential(1.0, (trials, n_train + 1))
    th = ca_alpha(n_train, pfa) * noise[:, 1:].mean(axis=1)
    return np.mean(noise[:, 0] > th)


def _train_idx(i, n, guard, train):
    """Training cells left and right of cell i, without wrapping: at the edges the window uses what exists
    on the other side (range profiles do not wrap around)."""
    left = [j for j in range(i - guard - train, i - guard) if 0 <= j < n]
    right = [j for j in range(i + guard + 1, i + guard + train + 1) if 0 <= j < n]
    if len(left) < train:
        right = [j for j in range(i + guard + 1, i + guard + 1 + 2 * train - len(left)) if j < n]
    if len(right) < train:
        left = [j for j in range(i - guard - (2 * train - len(right)), i - guard) if j >= 0]
    return left + right


def ca_cfar_edges(p, guard=2, train=8, pfa=1e-4):
    N = 2 * train; alpha = ca_alpha(N, pfa)
    n = len(p); th = np.array([alpha * np.mean(p[_train_idx(i, n, guard, train)]) for i in range(n)])
    return p > th, th


def os_cfar_1d(p, guard=2, train=8, k=None, pfa=1e-4, alpha=None):
    """Ordered-statistic CFAR: threshold from the k-th smallest training cell. alpha found by Monte Carlo
    for the given pfa if not supplied."""
    N = 2 * train; k = k or int(0.75 * N)
    if alpha is None:
        alpha = os_alpha(N, k, pfa)
    n = len(p); out = np.zeros(n, bool); th = np.full(n, np.nan)
    for i in range(n):
        idx = _train_idx(i, n, guard, train)
        th[i] = alpha * np.sort(p[idx])[k - 1]
        out[i] = p[i] > th[i]
    return out, th


def os_alpha(N, k, pfa, trials=400000, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.exponential(1.0, (trials, N + 1))
    ref = np.sort(z[:, 1:], axis=1)[:, k - 1]
    ratio = z[:, 0] / ref
    return float(np.quantile(ratio, 1 - pfa))


def range_profile(n=256, seed=4):
    """Power profile (linear) of one Doppler row: noise floor rising at short range (leakage/clutter),
    a clutter edge, three targets two of which are close."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    floor = 1 + 300 * np.exp(-x / 6) + np.where((x > 150) & (x < 200), 30, 0)
    p = floor * rng.exponential(1.0, n)
    for pos, snr_db in ((60, 18), (66, 15), (175, 20)):
        p[pos] += floor[pos] * 10 ** (snr_db / 10)
    return p, floor


class KalmanCV:
    """2-D constant-velocity Kalman filter: state [x, y, vx, vy]."""
    def __init__(self, z0, dt, q=2.0, r=0.15):
        self.dt = dt
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = dt
        self.H = np.zeros((2, 4)); self.H[0, 0] = self.H[1, 1] = 1
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.Q = q ** 2 * g @ g.T
        self.R = r ** 2 * np.eye(2)
        self.x = np.array([z0[0], z0[1], 0, 0], float)
        self.P = np.diag([r ** 2, r ** 2, 1.0, 1.0])

    def predict(self):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def gate(self, z, g=9.21):    # chi-square 2 dof, 99 %
        S = self.H @ self.P @ self.H.T + self.R
        d = z - self.H @ self.x
        return float(d @ np.linalg.solve(S, d)) < g, S

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


def walk_scene(n=160, dt=0.05, seed=2, pd=0.85, n_fa=1.5, r_meas=0.15):
    """A person walking an arc in front of the radar; detections with noise, misses and clutter false alarms."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    truth = np.c_[1.6 * np.sin(2 * np.pi * t / (n * dt)), 2.5 + 0.8 * np.cos(2 * np.pi * t / (n * dt))]
    dets = []
    for k in range(n):
        d = []
        if rng.random() < pd:
            d.append(truth[k] + rng.normal(0, r_meas, 2))
        for _ in range(rng.poisson(n_fa)):
            d.append(np.array([rng.uniform(-3, 3), rng.uniform(0.3, 5)]))
        dets.append(np.array(d).reshape(-1, 2))
    return t, truth, dets


def track(dets, dt=0.05, confirm=(4, 5), drop=4):
    """Track management: every unused detection starts a tentative track; a track is confirmed after
    `confirm` = M hits in the last N frames and dropped after `drop` consecutive misses. Each frame every
    track takes the nearest detection inside its 99 % gate. Returns the estimate of the confirmed track
    (NaN while none is confirmed), the detection it used, and the number of tracks ever confirmed."""
    M, N = confirm
    tracks = []; est = []; used = []; n_conf = 0
    for d in dets:
        free = list(range(len(d)))
        for tr in tracks:
            tr["kf"].predict()
            best = None
            for j in free:
                ok, _ = tr["kf"].gate(d[j])
                if ok:
                    dist = np.sum((d[j] - tr["kf"].x[:2]) ** 2)
                    if best is None or dist < best[0]: best = (dist, j)
            if best is not None:
                tr["kf"].update(d[best[1]]); free.remove(best[1]); tr["hist"].append(1); tr["miss"] = 0
                tr["z"] = d[best[1]]
            else:
                tr["hist"].append(0); tr["miss"] += 1; tr["z"] = None
            if not tr["conf"] and sum(tr["hist"][-N:]) >= M:
                tr["conf"] = True; n_conf += 1
        tracks = [t for t in tracks if t["miss"] < (drop if t["conf"] else 2)]
        for j in free:
            tracks.append(dict(kf=KalmanCV(d[j], dt), hist=[1], miss=0, conf=False, z=d[j]))
        conf = [t for t in tracks if t["conf"]]
        if conf:
            t = max(conf, key=lambda t: sum(t["hist"]))
            est.append(t["kf"].x[:2].copy()); used.append(t["z"])
        else:
            est.append([np.nan, np.nan]); used.append(None)
    return np.array(est), used, n_conf


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    for t_db in (10, 13, 15):
        print(f"fixed threshold {t_db} dB over the noise mean: Pfa {pfa_fixed(10**(t_db/10)):.2e}")
    print(f"threshold for Pfa 1e-6: {10*np.log10(-np.log(1e-6)):.2f} dB over noise mean")
    for N in (8, 16, 32):
        a = ca_alpha(N, 1e-6)
        print(f"CA-CFAR N={N}: alpha {a:.2f} ({10*np.log10(a):.2f} dB), MC Pfa at 1e-3 target "
              f"{mc_ca_pfa(N, 1e-3):.2e}")
    base = snr_needed(lambda s: pd_swerling1_known(s, 1e-6))
    for N in (8, 16, 32):
        ca = snr_needed(lambda s: pd_swerling1_ca(s, 1e-6, N))
        print(f"Swerling 1, Pd 0.9, Pfa 1e-6: known noise {base:.2f} dB, CA N={N} {ca:.2f} dB, loss {ca-base:.2f} dB")
    print(f"Albersheim Pd 0.9 Pfa 1e-6 N=1: {albersheim(0.9, 1e-6):.2f} dB; N=16 non-coherent: {albersheim(0.9,1e-6,16):.2f} dB")
    for cells, fps, pfa in ((256 * 64, 20, 1e-6), (256 * 64, 20, 1e-4), (128 * 32, 10, 1e-6)):
        print(f"{cells} cells x {fps} fps, Pfa {pfa:g}: {cells*fps*pfa:.2f} false alarms per second")
    p, floor = range_profile()
    d1, th1 = ca_cfar_edges(p, 2, 8, 1e-4)
    d2, th2 = os_cfar_1d(p, 2, 8, pfa=1e-4)
    print("CA detections", np.flatnonzero(d1), " OS detections", np.flatnonzero(d2))
    t, truth, dets = walk_scene()
    est, used, n_conf = track(dets)
    ok = ~np.isnan(est[:, 0])
    err = np.linalg.norm(est[ok] - truth[ok], axis=1)
    print(f"first confirmed at frame {np.argmax(ok)}, tracks ever confirmed {n_conf}")
    raw = [np.linalg.norm(u - tr) for u, tr in zip(used, truth) if u is not None]
    print(f"tracker RMS error {np.sqrt(np.mean(err**2)):.3f} m vs raw detection RMS {np.sqrt(np.mean(np.square(raw))):.3f} m; "
          f"detections in scene {sum(len(d) for d in dets)}")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(10 * np.log10(p), lw=1, label="range profile"); ax[0].plot(10 * np.log10(th1), label="CA-CFAR")
        ax[0].plot(10 * np.log10(th2), label="OS-CFAR"); ax[0].set_xlabel("range bin"); ax[0].set_ylabel("dB"); ax[0].legend()
        allp = np.array([q for d in dets for q in d])
        ax[1].plot(allp[:, 0], allp[:, 1], ".", color="0.7", label="detections (incl. false alarms)")
        ax[1].plot(truth[:, 0], truth[:, 1], "k--", label="truth"); ax[1].plot(est[:, 0], est[:, 1], "r", label="Kalman track")
        ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y (m)"); ax[1].legend(); fig.tight_layout()
        show(fig, "ch16_cfar_tracking.png")
