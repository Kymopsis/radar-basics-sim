"""Unit checks for radar_sim.pipeline: CFAR false-alarm rate, single-cell Doppler DFT, angle estimators, tracker."""
import numpy as np
from radar_sim import Radar, Target, cube, rd_map, range_fft
from radar_sim.pipeline import (ca_cfar_2d, peaks_2d, doppler_dft_at, angle_dft_at, angle_split_phase,
                                steering, op_counts, Tracker)

POS = np.arange(16) * 0.5
COARSE = np.degrees(np.arcsin((np.arange(16) - 7.5) / 8))


def test_cfar_2d_false_alarm_rate():
    rng = np.random.default_rng(0)
    p = rng.exponential(1.0, (200, 400))
    hits, _ = ca_cfar_2d(p, guard=(1, 1), train=(3, 3), pfa=1e-3)
    assert 0.5e-3 < hits.mean() < 2e-3


def test_cfar_finds_targets_and_doppler_dft_matches():
    r = Radar(rx=np.c_[POS, np.zeros(16)], n_samples=128, n_chirps=32, noise_dbfs=-20)
    c = cube(r, [Target(3.0, 2.0, 10), Target(6.0, -1.0, -25)])
    m, ranges, vel, rd = rd_map(r, c)
    det = np.argwhere(peaks_2d(m ** 2, ca_cfar_2d(m ** 2)[0]))
    assert len(det) == 2
    rc = range_fft(c)
    for d, rr in det:
        assert np.allclose(doppler_dft_at(rc, rr, d), rd[d, :, rr])
        ang, _ = angle_split_phase(rd[d, :, rr], POS, COARSE)
        assert min(abs(ang - 10), abs(ang + 25)) < 0.5


def test_split_phase_and_dft_accuracy_high_snr():
    for th in (-37.0, -3.3, 0.0, 21.7, 40.0):
        s = steering(POS, th)[0]
        assert abs(angle_split_phase(s, POS, COARSE)[0] - th) < 1e-6
        assert abs(angle_dft_at(s, POS, np.linspace(-60, 60, 481))[0] - th) <= 0.125 + 1e-9


def test_op_counts_order():
    o = op_counts(64, 128, 16, 30)
    assert o["split_phase_at_detections"] < o["dft_at_detections"] < o["full_3d_fft"]


def test_tracker_confirms_and_ignores_clutter():
    rng = np.random.default_rng(1)
    trk = Tracker(0.1, confirm=(3, 5), drop=4)
    for k in range(30):
        truth = np.array([-1 + 0.05 * k, 3.0])
        dets = [truth + rng.normal(0, 0.05, 2), rng.uniform([-3, 0.5], [3, 6])]
        out = trk.step(dets)
    assert len(out) >= 1
    best = min(out, key=lambda t: np.hypot(*(t[1][:2] - truth)))
    assert np.hypot(*(best[1][:2] - truth)) < 0.1 and abs(best[1][2] - 0.5) < 0.2


def test_beam_monopulse_and_tdm_comp():
    from radar_sim.pipeline import beam_monopulse, tdm_phase_comp
    for th in (-38.0, -3.3, 0.0, 21.7):
        assert abs(beam_monopulse(steering(POS, th)[0], POS, np.arange(-40, 40.1, 5.0))[0] - th) < 1e-6
    s = steering(POS, 10.0)[0]
    k = np.repeat(np.arange(2), 8)
    moved = s * np.exp(2j * np.pi * (40 - 32) * k / (64 * 2))
    assert np.allclose(tdm_phase_comp(moved, 40, 64, 2, 8), s)
