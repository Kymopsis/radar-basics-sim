"""Chapter 7: radar equation for an FMCW radar. SNR after the range FFT and after integrating N chirps,
and the range at which a target reaches a required SNR. All parameters are command-line options."""
import numpy as np
from _common import args, show
from radar_sim import C

K, T0 = 1.380649e-23, 290.0
a = args(__doc__)
for name, val, hlp in [("pt-dbm", 10.0, "transmit power"), ("g-dbi", 8.0, "TX and RX antenna gain"), ("nf-db", 12.0, "noise figure"),
                       ("loss-db", 3.0, "other losses"), ("fc", 60.5e9, "carrier (Hz)"), ("tc", 40e-6, "sampling time per chirp (s)"),
                       ("rcs", 1.0, "radar cross-section (m^2)"), ("chirps", 64, "chirps integrated"), ("need-db", 13.0, "required SNR")]:
    a.add_argument("--" + name, type=float, default=val, help=hlp)
o = a.parse_args()
lam = C / o.fc; db = lambda x: 10 ** (x / 10)

def snr(R, n=1):
    s = db(o.pt_dbm - 30) * db(o.g_dbi) ** 2 * lam ** 2 * o.rcs * n * o.tc / ((4 * np.pi) ** 3 * R ** 4 * K * T0 * db(o.nf_db) * db(o.loss_db))
    return 10 * np.log10(s)

for R in (1, 3, 10, 30):
    print(f"R = {R:>4} m: SNR one chirp {snr(R):6.1f} dB, {int(o.chirps)} chirps {snr(R, o.chirps):6.1f} dB")
Rreq = (db(snr(1.0, o.chirps) - o.need_db)) ** 0.25
print(f"range where SNR = {o.need_db:.0f} dB after {int(o.chirps)} chirps: {Rreq:.1f} m (textbook upper bound)")
if not o.no_plot:
    import matplotlib.pyplot as plt
    R = np.logspace(-0.5, 1.7, 300); fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogx(R, snr(R), label="one chirp"); ax.semilogx(R, snr(R, o.chirps), label=f"{int(o.chirps)} chirps")
    ax.axhline(o.need_db, ls=":"); ax.set_xlabel("range (m)"); ax.set_ylabel("SNR (dB)"); ax.legend()
    fig.tight_layout(); show(fig, "ch07_snr.png")
