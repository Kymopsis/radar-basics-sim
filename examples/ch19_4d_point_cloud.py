"""Chapter 19 - 4D imaging radar: a large TDM-MIMO array, range/Doppler/azimuth/elevation, point clouds.

Run:  python examples/ch19_4d_point_cloud.py  [--no-plot]
  1. builds a 12TX/16RX cascade-style layout and its virtual array. The positions are our own reconstruction,
     chosen to reproduce the properties TI publishes for its 4-chip cascade reference design (12 TX, 16 RX,
     3 TX offset for elevation, an 86-element lambda/2 azimuth row); they are not copied from TI files.
  2. simulates a 77 GHz road scene (car rear, pedestrian, overhead sign, guard-rail posts)
  3. range FFT -> Doppler FFT -> CA-CFAR on the range-Doppler map -> TDM motion compensation ->
     2-D (azimuth/elevation) beamforming per detection -> point cloud (x, y, z, v, power)
Textbook model, not any product. Uses radar_sim.
"""
import numpy as np
from _common import args, show
from radar_sim.fmcw import Radar, Target, cube, range_fft, doppler_fft, C

H = 0.5   # lambda/2 in lambda units
TX_AZ = np.array([11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0]) * H
TX_EL = np.array([6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * H
RX_AZ = np.array(list(range(11, 15)) + list(range(50, 54)) + list(range(46, 50)) + list(range(0, 4))) * H


def layout():
    tx = np.c_[TX_AZ, TX_EL]; rx = np.c_[RX_AZ, np.zeros(16)]
    return tx, rx


def radar(n_chirps=32):
    tx, rx = layout()
    return Radar(fc=77e9, bw=1e9, tc=20e-6, fs=12.8e6, n_samples=256, n_chirps=n_chirps, t_idle=5e-6,
                 tx=tx, rx=rx, noise_dbfs=-70)


def scene():
    """Targets as point scatterers; radar at the origin, 0.5 m above the road, y = forward."""
    pts = []
    def add(x, y, z, v, amp):
        r = np.sqrt(x * x + y * y + (z - 0.5) ** 2)
        az = np.degrees(np.arctan2(x, y)); el = np.degrees(np.arcsin((z - 0.5) / r))
        pts.append((Target(r=r, v=v, az=az, el=el, amp=amp), (x, y, z)))
    for x in (-0.8, -0.3, 0.3, 0.8):                     # car rear, closing at 1.5 m/s, 18 m ahead
        for z, dy in ((0.4, 0.0), (0.9, 0.3)):            # bumper line, then boot/lamp line 30 cm further
            add(x + 0.3, 18.0 + dy, z, -1.5, 1.0)
    for x in (-0.6, 1.2):                                # rear wheels, 0.6 m further
        add(x, 18.6, 0.3, -1.5, 0.7)
    add(0.3, 18.0, 1.3, -1.5, 0.5)                       # roof edge
    add(-3.5, 11.0, 0.9, 1.0, 0.25)                      # pedestrian walking away, left kerb
    add(-3.5, 11.0, 1.5, 1.0, 0.15)
    for x in (-2.0, 0.0, 2.0):                           # overhead sign gantry 30 m ahead, 5.5 m up (static)
        add(x, 30.0, 5.5, 0.0, 0.8)
    for y in (6, 10, 14, 22, 26):                        # guard-rail posts on the right, static
        add(4.0, y, 0.6, 0.0, 0.3)
    return pts


def unique_virtual(v):
    return {(round(a, 6), round(b, 6)) for a, b in v}


def ca_cfar_2d(p, guard=(2, 2), train=(4, 8), pfa=1e-6):
    """2-D CA-CFAR on a power map [doppler, range] (wraps in Doppler, not in range)."""
    from numpy.lib.stride_tricks import sliding_window_view
    gd, gr = guard; td, tr = train
    kd, kr = gd + td, gr + tr
    pad = np.pad(p, ((kd, kd), (0, 0)), mode="wrap")
    pad = np.pad(pad, ((0, 0), (kr, kr)), mode="edge")
    win = sliding_window_view(pad, (2 * kd + 1, 2 * kr + 1))
    tot = win.sum(axis=(-1, -2))
    inner = win[..., td:td + 2 * gd + 1, tr:tr + 2 * gr + 1].sum(axis=(-1, -2))
    n = (2 * kd + 1) * (2 * kr + 1) - (2 * gd + 1) * (2 * gr + 1)
    alpha = n * (pfa ** (-1 / n) - 1)
    th = alpha * (tot - inner) / n
    det = p > th
    # keep local maxima only
    from scipy.ndimage import maximum_filter
    return det & (p == maximum_filter(p, size=3, mode="wrap"))


def point_cloud(r, c, az_grid=np.arange(-60, 60.01, 0.25), el_grid=np.arange(-20, 20.01, 0.25), max_per_cell=6,
                rel_db=12):
    """Two-step angle estimation per detected range-Doppler cell (a common approach for arrays with a dense
    azimuth row and a few sparse elevated rows):
      1. azimuth spectrum from the elements of the z = 0 row (overlapping positions averaged); keep every local
         peak within rel_db of the strongest, up to max_per_cell (several reflectors can share one cell)
      2. for each azimuth peak, remove the azimuth phase from all virtual channels, average each row, and run a
         small elevation spectrum over the 4 row values (z = 0, 0.5, 2, 3 lambda), equal weight per row."""
    from scipy.ndimage import maximum_filter1d
    rd = doppler_fft(range_fft(c))                                   # [dopp, virt, range]
    p = (np.abs(rd) ** 2).sum(axis=1)
    det = ca_cfar_2d(p)
    det[:, :4] = False                                                # skip the first bins (leakage zone)
    virt = r.virtual(); n_tx, n_rx = len(r.tx), len(r.rx)
    k_tx = np.repeat(np.arange(n_tx), n_rx)
    row0 = np.abs(virt[:, 1]) < 1e-9
    xs = np.unique(np.round(virt[row0, 0], 6))
    rows = np.unique(np.round(virt[:, 1], 6))
    Saz = np.exp(2j * np.pi * np.outer(np.sin(np.radians(az_grid)), xs))
    Sel = np.exp(2j * np.pi * np.outer(np.sin(np.radians(el_grid)), rows))
    out = []
    for iv, ir in zip(*np.nonzero(det)):
        snap = rd[iv, :, ir]
        dpsi = 2 * np.pi * (iv - r.n_chirps // 2) / r.n_chirps       # TDM motion compensation (ch 15)
        snap = snap * np.exp(-1j * dpsi * k_tx / n_tx)
        # 1. azimuth from the z = 0 row
        x0 = np.round(virt[row0, 0], 6); s0 = snap[row0]
        u = np.array([s0[x0 == x].mean() for x in xs]) * np.hanning(len(xs) + 2)[1:-1]
        paz = np.abs(Saz.conj() @ u) ** 2
        loc = (paz == maximum_filter1d(paz, 9, mode="nearest")) & (paz > paz.max() * 10 ** (-rel_db / 10))
        cand = sorted(np.nonzero(loc)[0], key=lambda q: -paz[q])[:max_per_cell]
        rng_ = ir * r.range_res; vel = (iv - r.n_chirps // 2) * r.v_res
        for ia in cand:
            az = np.radians(az_grid[ia])
            # 2. elevation from the row values after removing this azimuth's phase
            rem = snap * np.exp(-2j * np.pi * virt[:, 0] * np.sin(az))
            rv = np.array([rem[np.abs(virt[:, 1] - z) < 1e-6].mean() for z in rows])
            pel = np.abs(Sel.conj() @ rv) ** 2
            el = np.radians(el_grid[np.argmax(pel)])
            x = rng_ * np.cos(el) * np.sin(az); y = rng_ * np.cos(el) * np.cos(az); z = 0.5 + rng_ * np.sin(el)
            out.append((x, y, z, vel, 10 * np.log10(p[iv, ir] * paz[ia] / paz.max())))
    return np.array(out), p, det


if __name__ == "__main__":
    o = args(__doc__).parse_args()
    tx, rx = layout(); r = radar()
    v = r.virtual(); uv = unique_virtual(v)
    az_row = sorted({a for a, b in uv if abs(b) < 1e-9})
    print(f"virtual channels {len(v)}, unique positions {len(uv)}; azimuth row {len(az_row)} elements "
          f"from {az_row[0]} to {az_row[-1]} lambda, contiguous: {np.allclose(np.diff(az_row), 0.5)}")
    print("elevation rows (lambda):", sorted({b for a, b in uv}))
    print(f"azimuth resolution ~{np.degrees(2/len(az_row)):.2f} deg; elevation aperture {max(TX_EL):.1f} lambda "
          f"-> ~{np.degrees(1/max(TX_EL)):.0f} deg (sparse, coarse)")
    print(r.summary(), f" TX repeat {r.t_frame_chirp*1e6:.0f} us")
    pts = scene()
    c = cube(r, [t for t, _ in pts], seed=1)
    pc, p, det = point_cloud(r, c)
    print(f"{len(pc)} detections -> points")
    for x, y, z, vv, pw in sorted(pc.tolist(), key=lambda q: q[1]):
        print(f"  x {x:+6.2f}  y {y:6.2f}  z {z:5.2f}  v {vv:+5.2f}  P {pw:6.1f} dB")
    if not o.no_plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(projection="3d")
        tr = np.array([q for _, q in pts])
        ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2], marker="x", c="0.6", label="true reflectors")
        sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 3], cmap="coolwarm", vmin=-2, vmax=2, s=60, label="radar points")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="radial velocity (m/s)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)"); ax.legend()
        show(fig, "ch19_point_cloud.png")
