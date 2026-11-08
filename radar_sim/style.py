"""Shared figure + animation style for the Kymopsis Radar Basics series.
Every chapter imports this so all figures and loops look the same."""
import os, subprocess, tempfile, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_FONT_DIR = os.environ.get("RADAR_SIM_FONT_DIR", "/usr/share/fonts/opentype/inter/")
for f in ("Inter-Regular.otf", "Inter-Medium.otf", "Inter-SemiBold.otf", "Inter-Bold.otf"):
    p = _FONT_DIR + f
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 15, "axes.edgecolor": "#9aa3b2",
    "axes.labelcolor": "#1a2230", "xtick.color": "#4a5263", "ytick.color": "#4a5263",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e9ee", "grid.linewidth": 1, "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff", "image.cmap": "viridis",
})
INK, MUTED = "#1a2230", "#6b7484"
C1, C2, C3, C4 = "#0e8fa8", "#7d55c7", "#d9682b", "#2f9e5b"
TAG = "SIMULATION"


def head(fig, title, sub=None):
    fig.text(0.02, 0.94, title, fontsize=26, weight="bold", color=INK, va="top")
    if sub:
        fig.text(0.02, 0.875, sub, fontsize=16, color=MUTED, va="top")


def foot(fig, text, sim=True):
    t = (f"{TAG} · " if sim else "") + text + "  ·  kymopsis.com"
    fig.text(0.02, 0.02, t, fontsize=12, color=MUTED)


def save_png(fig, path, dpi=100):
    """16x9 in at dpi 100 = 1600x900 px."""
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


def save_loop(draw, n_frames, path, fps=24, size=(12.0, 6.75), dpi=100, poster_frame=0, crf=30):
    """Render a short silent looping clip. draw(fig, i) must clear+redraw the figure for frame i.
    Writes <path>.mp4 (H.264, yuv420p, faststart, no audio, 1200x675) and <path>-poster.jpg.
    Blog limits ([kym_anim] shortcode): 4-12 s seamless loop, 24-30 fps, <=1 MB, <=5 clips per post,
    poster <=150 KB. Make frame n_frames equal frame 0 so the loop is seamless."""
    tmp = tempfile.mkdtemp()
    fig = plt.figure(figsize=size)
    for i in range(n_frames):
        fig.clf()
        draw(fig, i)
        fig.savefig(os.path.join(tmp, f"f{i:04d}.png"), dpi=dpi)
        if i == poster_frame:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(tmp, f"f{i:04d}.png"),
                            "-q:v", "4", path + "-poster.jpg"], check=True)
    plt.close(fig)
    out = path + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(tmp, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(crf), "-preset", "slow", "-movflags", "+faststart", "-an",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
    shutil.rmtree(tmp)
    kb = os.path.getsize(out) // 1024
    print("wrote", out, kb, "KB")
    if kb > 1024:
        print("WARNING: over 1 MB, raise crf or shorten the clip")
    return out
