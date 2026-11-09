"""Chapter 12: angle of arrival from the phase difference between receive channels (1T4R)."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, rd_map, angle_spectrum

o = args(__doc__).parse_args()
r = make("1T4R")
for az in [-40, -10, 0, 25]:
    m, rr, vv, rd = rd_map(r, cube(r, [Target(3.0, 0.5, az=az)]))
    iv, ir = np.unravel_index(np.argmax(m), m.shape)
    snap = rd[iv, :, ir]
    dphi = np.degrees(np.angle(snap[1] / snap[0]))
    est = np.degrees(np.arcsin(np.radians(dphi) / np.pi))
    ang = np.linspace(-90, 90, 1801); sp = angle_spectrum(snap, r.virtual()[:, 0], ang)
    print(f"true {az:+d} deg: adjacent phase diff {dphi:+.1f} deg -> {est:+.1f} deg; FFT/Bartlett peak {ang[np.argmax(sp)]:+.1f} deg")
