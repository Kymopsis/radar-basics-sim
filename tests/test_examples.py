"""Run chapter examples headless and check the numbers they print against the known truth."""
import os, re, subprocess, sys
import pytest

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


def run(name, *extra):
    out = subprocess.run([sys.executable, os.path.join(EX, name), "--no-plot", *extra],
                         capture_output=True, text=True, check=True, env={**os.environ, "MPLBACKEND": "Agg"})
    return out.stdout


@pytest.mark.parametrize("vt", [30.0, -22.5, 7.3])
def test_ch16_two_interval_unfolding(vt):
    m = re.search(r"unfolded ([-+\d.]+) m/s", run("ch16_ambiguity.py", "--vt", str(vt)))
    assert abs(float(m.group(1)) - vt) < 0.01


@pytest.mark.parametrize("v", [9.0, -9.0, 4.0])
def test_ch16_tdm_fold(v):
    m = re.search(r"chosen ([-+\d.]+) m/s at ([-+\d.]+) deg", run("ch16_ambiguity.py", "--v", str(v)))
    assert abs(float(m.group(1)) - v) < 0.5 and abs(float(m.group(2)) - 20.0) < 1.0


def test_ch02_phase_displacement():
    m = re.search(r"noise-free model\) ([\d.]+) um", run("ch02_phase_iq.py"))
    assert float(m.group(1)) < 1.0
