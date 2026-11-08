import numpy as np
from radar_sim import make, Target, cube, rd_map, angle_spectrum, music, C


def test_range_velocity_angle():
    r = make("1T4R")
    c = cube(r, [Target(r=3.0, v=1.0, az=20.0)])
    m, rr, vv, rd = rd_map(r, c)
    iv, ir = np.unravel_index(np.argmax(m), m.shape)
    assert abs(rr[ir] - 3.0) <= r.range_res
    assert abs(vv[iv] - 1.0) <= r.v_res
    ang = np.linspace(-90, 90, 721)
    sp = angle_spectrum(rd[iv, :, ir], r.virtual()[:, 0], ang)
    assert abs(ang[np.argmax(sp)] - 20) < 3


def test_formulas():
    r = make("1T2R", bw=4e9, n_samples=256, fs=6.4e6, tc=40e-6)
    assert abs(r.range_res - C / (2 * 4e9 * (256 / 6.4e6) / 40e-6)) < 1e-12
    assert abs(r.lam - C / r.fc) < 1e-15


def test_virtual_array_size():
    r = make("2T4R")
    v = r.virtual()[:, 0]
    assert len(v) == 8 and np.allclose(np.diff(np.sort(v)), 0.5)


def test_music_resolves_close_targets():
    pos = np.arange(8) * 0.5
    rng = np.random.default_rng(1)
    th = np.radians([0, 8])
    A = np.exp(1j * 2 * np.pi * np.outer(pos, np.sin(th)))
    S = (rng.standard_normal((2, 200)) + 1j * rng.standard_normal((2, 200))) / np.sqrt(2)
    X = A @ S + 0.05 * (rng.standard_normal((8, 200)) + 1j * rng.standard_normal((8, 200)))
    R = X @ X.conj().T / 200
    ang = np.linspace(-30, 30, 1201)
    p = music(R, pos, ang, 2)
    peaks = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] > p[i + 1] and p[i] > 0.1]
    assert len(peaks) >= 2
