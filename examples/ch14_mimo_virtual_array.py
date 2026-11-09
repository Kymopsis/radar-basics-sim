"""Chapter 14: TDM-MIMO virtual array — 2T4R gives 8 virtual channels, beam twice as narrow as 1T4R."""
import numpy as np
from _common import args, show
from radar_sim import make, Target, cube, rd_map, angle_spectrum

o = args(__doc__).parse_args()
for lay in ["1T4R", "2T4R"]:
    r = make(lay, n_chirps=64)
    m, rr, vv, rd = rd_map(r, cube(r, [Target(3.0, 0.0, az=0.0)], motion=False))
    iv, ir = np.unravel_index(np.argmax(m), m.shape)
    ang = np.linspace(-90, 90, 3601); sp = angle_spectrum(rd[iv, :, ir], r.virtual()[:, 0], ang)
    hp = ang[sp >= sp.max() / np.sqrt(2)]
    print(f"{lay}: {len(r.virtual())} virtual elements, -3 dB beamwidth {hp.max()-hp.min():.1f} deg")
