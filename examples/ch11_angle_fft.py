"""Old name, kept so earlier links and commands keep working: since the 2026-11-09 renumbering of
Radar Principles this example is ch12_angle_fft.py. This file only runs it, with the same arguments."""
import pathlib, runpy, sys

new = pathlib.Path(__file__).resolve().with_name("ch12_angle_fft.py")
sys.argv[0] = str(new)
runpy.run_path(str(new), run_name="__main__")
