"""Rebuild notebooks/quickstart.ipynb and execute it in-process (stdout + PNG outputs embedded): python tools/build_notebook.py"""
import json, io, base64, contextlib, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ".")
cells = [
("md", "# Radar Principles: quick start\n\nA guided tour of `radar_sim`, the companion code for the [Radar Principles](https://kymopsis.com/series/radar-principles/) series. "
       "One simulated 60 GHz FMCW radar, followed from raw samples to range, velocity and angle. Each step links to the chapter that explains it.\n\n"
       "Textbook point-target model, not any product. Run `pip install numpy matplotlib scipy` first."),
("code", "import numpy as np\nimport matplotlib.pyplot as plt\nfrom radar_sim import make, Target, cube, range_fft, rd_map, angle_spectrum\n\n"
         "radar = make(\"2T4R\", fc=60.5e9, bw=4e9, n_chirps=64)   # TDM-MIMO: 2 TX x 4 RX = 8 virtual channels\nprint(radar.summary())"),
("md", "## 1. A scene and its data cube\n\nThree point targets. `cube` returns the dechirped I/Q samples as `[chirp, virtual channel, sample]` "
       "(chapters 5 and 19)."),
("code", "targets = [Target(r=1.2, v=0.0, az=-20), Target(r=2.5, v=0.8, az=10), Target(r=3.4, v=-0.5, az=35, amp=0.6)]\n"
         "data = cube(radar, targets)\nprint(data.shape, data.dtype)"),
("md", "## 2. Range FFT\n\nOne FFT per chirp turns beat frequency into range; the bin width is c/2B (chapter 6)."),
("code", "rp = np.abs(range_fft(data))[0, 0]\nr_axis = np.arange(len(rp)) * radar.range_res\n"
         "plt.figure(figsize=(8, 3)); plt.plot(r_axis, 20 * np.log10(rp + 1e-12)); plt.xlim(0, 5)\n"
         "plt.xlabel(\"range (m)\"); plt.ylabel(\"dB\"); plt.title(\"range profile, one chirp\"); plt.show()"),
("md", "## 3. Range-Doppler map\n\nA second FFT across chirps gives radial velocity (chapter 10)."),
("code", "m, ranges, velocities, rd = rd_map(radar, data)\n"
         "plt.figure(figsize=(8, 4)); plt.imshow(10 * np.log10(m + 1e-12), aspect=\"auto\", origin=\"lower\",\n"
         "    extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]], vmin=-40)\n"
         "plt.xlim(0, 5); plt.xlabel(\"range (m)\"); plt.ylabel(\"velocity (m/s)\"); plt.colorbar(label=\"dB\"); plt.show()"),
("md", "## 4. Angle from the virtual channels\n\nFor each peak cell, the 8 channel values form a snapshot; an FFT/Bartlett scan across them gives angle "
       "(chapters 12 and 14). Note: with TDM-MIMO, a moving target needs Doppler compensation first (chapter 15); the static target is exact."),
("code", "ang = np.linspace(-90, 90, 721)\nfor t in targets:\n"
         "    ir = int(np.argmin(np.abs(ranges - t.r))); iv = int(np.argmin(np.abs(velocities - t.v)))\n"
         "    sp = angle_spectrum(rd[iv, :, ir], radar.virtual()[:, 0], ang)\n"
         "    print(f\"target at {t.r} m, {t.v:+.1f} m/s, true {t.az:+d} deg -> peak {ang[np.argmax(sp)]:+.1f} deg\")"),
("md", "## Next\n\nEach chapter has its own script in `examples/` (run with `--no-plot` for numbers only), e.g. "
       "`python examples/ch09_phase_precision.py` for micrometre ranging from phase, or `python examples/ch19_4d_point_cloud.py` "
       "for a full 4D point cloud."),
]
ns, nb_cells, ec = {}, [], 0
for kind, src in cells:
    lines = [l + "\n" for l in src.split("\n")]; lines[-1] = lines[-1].rstrip("\n")
    if kind == "md":
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines}); continue
    ec += 1; outs = []
    buf = io.StringIO(); figs = []
    plt.show = lambda *a, **k: figs.append(plt.gcf()) or None
    with contextlib.redirect_stdout(buf):
        exec(src, ns)
    if buf.getvalue():
        outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
    for f in figs:
        b = io.BytesIO(); f.savefig(b, format="png", dpi=80, bbox_inches="tight"); plt.close(f)
        outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
    nb_cells.append({"cell_type": "code", "execution_count": ec, "metadata": {}, "outputs": outs, "source": lines})
nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("notebooks/quickstart.ipynb", "w"), indent=1)
print("ok", os.path.getsize("notebooks/quickstart.ipynb"))
for c in nb_cells:
    for o in c.get("outputs", []):
        if o["output_type"] == "stream": print("".join(o["text"]))
