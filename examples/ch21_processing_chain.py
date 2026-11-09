"""Chapter 21 - the real-time processing chain: range-Doppler -> 2-D CFAR -> angle only at detections -> tracks.

Run:  python examples/ch21_processing_chain.py  [--no-plot]
  1. one frame of a 2T8R TDM-MIMO radar (16-channel lambda/2 virtual row): range FFT, Doppler FFT, 2-D CA-CFAR
     on the non-coherent range-Doppler power map, 3x3 peak picking, TDM Doppler phase compensation
  2. a second channel does not need its own Doppler FFT: a single-cell DFT over the chirps at each detection
     gives exactly the value the full 2-D FFT would have there (checked numerically)
  3. four ways to get the angle: FFT over the virtual channels for every cell ("3D FFT"), DFT only at
     detections, coarse subarray DFT + phase comparison of the two half arrays (all digital, MIMO), and an
     analog / AESA-style beam scan where only two half-array sums are digitised and the angle inside the
     chosen beam comes from their phase. RMS error versus SNR (Monte Carlo) and complex multiplies per frame
  4. 40 frames of two people walking: the whole chain feeds a constant-velocity Kalman tracker with gating
     and M-of-N confirmation
Textbook model, not any product. Uses radar_sim.
"""
import numpy as np
from _common import args, show
from radar_sim.fmcw import Radar, Target, cube, rd_map, range_fft
from radar_sim.pipeline import (ca_cfar_2d, peaks_2d, doppler_dft_at, angle_fft_all, angle_dft_at,
                                angle_split_phase, beam_monopulse, tdm_phase_comp, op_counts, steering,
                                Tracker, to_xy)

N_VIRT = 16
POS = np.arange(N_VIRT) * 0.5
FINE = np.linspace(-60, 60, 481)                                   # 0.25 deg grid for the DFT scan
COARSE = np.degrees(np.arcsin((np.arange(16) - 7.5) / 8))          # 16 beams, 0.125 apart in sin(theta)
DT = 0.1
BEAMS = np.arange(-40, 40.1, 5.0)                                   # analog beam positions, 5 deg steps
TX = np.array([[0.0, 0.0], [4.0, 0.0]]); RX = np.c_[np.arange(8) * 0.5, np.zeros(8)]   # 2T8R -> 16 virtual


def radar():
    return Radar(fc=60.5e9, bw=4e9, tc=40e-6, fs=6.4e6, n_samples=128, n_chirps=64, t_idle=200e-6,
                 tx=TX, rx=RX, noise_dbfs=15)


def frame(r, targets, seed):
    c = cube(r, targets, seed=seed)
    m, ranges, vel, rd = rd_map(r, c)
    p = m ** 2
    hits, _ = ca_cfar_2d(p, guard=(2, 2), train=(4, 8), pfa=1e-6)
    pk = peaks_2d(p, hits)
    pk[:, :3] = False                                               # skip the first range cells (leakage)
    return c, p, ranges, vel, rd, np.argwhere(pk)


def angle_mc(snr_db, trials=600, seed=3):
    """RMS angle error of the three methods for one target at a random angle in +-40 deg.
    snr_db = per-channel SNR at the detection cell (after range and Doppler integration)."""
    rng = np.random.default_rng(seed)
    err = {"fft64": [], "dft": [], "split": [], "aesa": []}
    _, fft_ang = angle_fft_all(np.zeros((1, N_VIRT, 1)), 64)
    for _ in range(trials):
        th = rng.uniform(-40, 40)
        s = steering(POS, th)[0] * np.exp(2j * np.pi * rng.random())
        s = s + 10 ** (-snr_db / 20) / np.sqrt(2) * (rng.standard_normal(N_VIRT) + 1j * rng.standard_normal(N_VIRT))
        spec, _ = angle_fft_all(s[None, :, None], 64)
        err["fft64"].append(fft_ang[int(np.argmax(spec[0, :, 0]))] - th)
        err["dft"].append(angle_dft_at(s, POS, FINE)[0] - th)
        err["split"].append(angle_split_phase(s, POS, COARSE)[0] - th)
        err["aesa"].append(beam_monopulse(s, POS, BEAMS)[0] - th)
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}


def walkers(t):
    """Two people: one walking diagonally away on the left, one walking towards the radar on the right."""
    va, vb = np.array([0.6, 0.45]), np.array([0.0, -0.6])
    return np.array([-1.8, 2.0]) + va * t, np.array([1.2, 5.5]) + vb * t, va, vb


def targets_at(t):
    out = []
    for p, v in zip(walkers(t)[:2], walkers(t)[2:]):
        r = float(np.hypot(*p)); az = float(np.degrees(np.arctan2(p[0], p[1])))
        vr = float(v @ p / r)
        out.append(Target(r=r, v=vr, az=az, amp=1.0))
    out.append(Target(r=2.0, v=0.0, az=-35, amp=0.6))              # a static object (chair, wall corner)
    return out


def run_tracking(r, n_frames=40):
    trk = Tracker(DT, confirm=(3, 5), drop=4, q=1.0, r=0.12)
    raw, tracks, truth = [], {}, []
    n_det = []
    for k in range(n_frames):
        t = k * DT
        _, p, ranges, vel, rd, det = frame(r, targets_at(t), seed=100 + k)
        n_det.append(len(det))
        z = []
        for d, rr in det:
            if abs(vel[d]) < r.v_res / 2:                                  # drop zero-Doppler (static) cells
                continue
            snap = tdm_phase_comp(rd[d, :, rr], d, r.n_chirps, 2, 8)
            ang, _ = angle_split_phase(snap, POS, COARSE)
            z.append(to_xy(ranges[rr], ang))
        raw.append(z)
        for tid, x in trk.step(z):
            tracks.setdefault(tid, []).append((k, x[:2].copy()))
        truth.append(walkers(t)[:2])
    return raw, tracks, np.array(truth), n_det


def track_rms(tracks, truth):
    errs = []
    for pts in tracks.values():
        for k, x in pts:
            errs.append(min(np.hypot(*(x - truth[k][0])), np.hypot(*(x - truth[k][1]))))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def raw_rms(raw, truth):
    errs = [min(np.hypot(*(z - truth[k][0])), np.hypot(*(z - truth[k][1])))
            for k, zs in enumerate(raw) for z in zs]
    errs = [e for e in errs if e < 1.0]                             # associated detections only
    return float(np.sqrt(np.mean(np.square(errs))))


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    r = radar()
    print(r.summary(), f"| {N_VIRT}-channel lambda/2 virtual row")

    # 1-2. one frame
    c, p, ranges, vel, rd, det = frame(r, targets_at(0.0), seed=1)
    print(f"frame: {p.shape[0]} Doppler x {p.shape[1]} range cells, CFAR Pfa 1e-6, {len(det)} detections")
    rc = range_fft(c)
    worst = 0.0
    for d, rr in det:
        z = doppler_dft_at(rc, rr, d)
        worst = max(worst, float(np.max(np.abs(z - rd[d, :, rr]) / np.abs(rd[d, :, rr]))))
        ang, th0 = angle_split_phase(tdm_phase_comp(rd[d, :, rr], d, r.n_chirps, 2, 8), POS, COARSE)
        print(f"  det range {ranges[rr]:5.2f} m  v {vel[d]:+5.2f} m/s  angle {ang:+6.2f} deg (coarse {th0:+5.1f})")
    print(f"single-cell Doppler DFT vs full 2-D FFT: max relative difference {worst:.1e}")

    # 3. angle accuracy and cost
    print("angle RMS error (deg) vs per-channel SNR at the detection, one target in +-40 deg"
          " (AESA: best of 17 beams, as if every beam had been visited in its own dwell):")
    acc = {}
    for snr in (0, 5, 10, 15, 20, 30):
        acc[snr] = angle_mc(snr)
        print(f"  SNR {snr:2d} dB: 64-pt FFT {acc[snr]['fft64']:5.2f}   DFT 0.25deg grid {acc[snr]['dft']:5.2f}   "
              f"split-phase {acc[snr]['split']:5.2f}   AESA beam+monopulse {acc[snr]['aesa']:5.2f}")
    ops = op_counts(64, 128, N_VIRT, 30)
    print("complex multiplies per frame for the angle step (64 Doppler x 128 range cells, 30 detections):")
    for k, v in ops.items():
        print(f"  {k:26s} {v:12,.0f}")
    print(f"  split-phase is {ops['full_3d_fft']/ops['split_phase_at_detections']:,.0f}x cheaper than the full 3D FFT")

    # 4. tracking
    raw, tracks, truth, n_det = run_tracking(r)
    print(f"tracking: {len(tracks)} confirmed tracks over 40 frames, detections/frame {np.mean(n_det):.1f}")
    print(f"tracker RMS error {track_rms(tracks, truth):.3f} m vs raw detection RMS {raw_rms(raw, truth):.3f} m")

    if not o.no_plot:
        import matplotlib.pyplot as plt
        from radar_sim.style import head, foot, C1, C2, C3, C4, MUTED, INK
        SUB = "2T8R TDM-MIMO (16 virtual channels), 60.5 GHz, 4 GHz sweep. Simulation, textbook point targets."

        def panel_rd(ax, fs=15):
            ax.imshow(10 * np.log10(p + 1e-12), aspect="auto", origin="lower", cmap="viridis",
                      extent=[ranges[0], ranges[-1], vel[0], vel[-1]])
            ax.plot(ranges[det[:, 1]], vel[det[:, 0]], "o", mfc="none", mec="#ff4d4d", ms=14, mew=2)
            ax.set_xlim(0, 7); ax.set_xlabel("range (m)"); ax.set_ylabel("velocity (m/s)")

        def panel_acc(ax, fs=12):
            s = sorted(acc)
            for key, col, lab in (("fft64", C3, "64-point FFT over channels"), ("dft", C2, "DFT at detection, 0.25° grid"),
                                  ("split", C1, "coarse 4-ch DFT + half-array phase"),
                                  ("aesa", C4, "analog beam (5° steps) + monopulse")):
                ax.semilogy(s, [acc[k][key] for k in s], "o-", color=col, lw=2.5, label=lab)
            ax.set_xlabel("per-channel SNR at the detection (dB)"); ax.set_ylabel("RMS angle error (deg)")
            ax.legend(frameon=False, fontsize=fs)

        def panel_cost(ax, fs=13):
            names = ["full 3D FFT\n(every cell)", "DFT at\ndetections", "split-phase at\ndetections"]
            vals = list(ops.values())
            ax.barh(names, vals, color=[C3, C2, C1]); ax.set_xscale("log")
            for i, v in enumerate(vals):
                ax.text(v * 1.2, i, f"{v:,.0f}", va="center", fontsize=fs, color=INK)
            ax.set_xlim(50, 3e7); ax.set_xlabel("complex multiplies per frame (angle step only)")

        def panel_tracks(ax, fs=12):
            for zs in raw:
                for z in zs:
                    ax.plot(z[0], z[1], ".", color="#b8bfcc", ms=6)
            ax.plot(truth[:, 0, 0], truth[:, 0, 1], "--", color=MUTED, lw=1.5)
            ax.plot(truth[:, 1, 0], truth[:, 1, 1], "--", color=MUTED, lw=1.5, label="true path")
            for i, pts in enumerate(tracks.values()):
                xy = np.array([x for _, x in pts])
                ax.plot(xy[:, 0], xy[:, 1], "-", color=[C1, C2, C4, C3][i % 4], lw=3, label=f"track {i+1}")
            ax.plot(0, 0, "^", color=INK, ms=12)
            ax.set_xlim(-3, 3); ax.set_ylim(0, 6.5)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(frameon=False, fontsize=fs, loc="upper left")

        # overview (README gallery)
        fig = plt.figure(figsize=(16, 10))
        head(fig, "The real-time chain: CFAR first, then angle only where something was found", SUB)
        for rect, fn, title in (([0.06, 0.50, 0.40, 0.31], panel_rd, "1  Range-Doppler map, CFAR detections circled"),
                                ([0.56, 0.50, 0.40, 0.31], panel_acc, "2  Angle accuracy"),
                                ([0.14, 0.12, 0.32, 0.28], panel_cost, "3  Cost: 64 x 128 cells, 30 detections"),
                                ([0.56, 0.12, 0.40, 0.28], panel_tracks, "4  Detections (grey) and confirmed tracks")):
            ax = fig.add_axes(rect); fn(ax); ax.set_title(title, loc="left", fontsize=15)
        foot(fig, "radar-basics-sim · examples/ch21_processing_chain.py")
        show(fig, "ch21_processing_chain.png")

        # the same four panels as single 16:9 figures (readable on a phone)
        singles = (("ch21_rd_cfar.png", panel_rd, "Range-Doppler map with CFAR detections",
                    "Non-coherent sum of 16 channels, 2-D CA-CFAR at Pfa 1e-6, 3x3 peaks (red circles)", [0.09, 0.13, 0.86, 0.68]),
                   ("ch21_angle_accuracy.png", panel_acc, "Angle accuracy at a detection",
                    "RMS error vs per-channel SNR, one target in ±40°, 600 trials per point", [0.09, 0.13, 0.86, 0.68]),
                   ("ch21_angle_cost.png", panel_cost, "Cost of the angle step",
                    "Complex multiplies per frame: 64 Doppler x 128 range cells, 30 detections", [0.18, 0.14, 0.76, 0.66]),
                   ("ch21_tracks.png", panel_tracks, "Detections and confirmed tracks",
                    "40 frames at 10 frames/s, two people walking; grey = detections, lines = tracks", [0.10, 0.12, 0.85, 0.70]))
        for name, fn, title, sub, rect in singles:
            fig = plt.figure(figsize=(16, 9))
            head(fig, title, sub)
            ax = fig.add_axes(rect); fn(ax, fs=15)
            foot(fig, "radar-basics-sim · examples/ch21_processing_chain.py")
            show(fig, name)
