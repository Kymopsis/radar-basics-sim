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


def test_ch08_phase_reaches_bound():
    for line in run("ch08_phase_precision.py").splitlines():
        m = re.search(r"phase std\s+([\d.]+) um \(bound\s+([\d.]+) um\)", line)
        if m:
            assert float(m.group(1)) < 1.3 * float(m.group(2))


def test_ch12_numbers():
    out = run("ch12_antenna_aperture.py")
    assert "20.0 mm square aperture:  20.9 dBi, beam  12.6 deg" in out
    assert "first sidelobe: -13.3 dB" in out


def test_ch13_half_wave():
    assert "half-wave wall 1.14 mm" in run("ch13_radome_wall.py")


def test_ch14_tracker_beats_raw():
    m = re.search(r"tracker RMS error ([\d.]+) m vs raw detection RMS ([\d.]+) m", run("ch14_cfar_tracking.py"))
    assert float(m.group(1)) < float(m.group(2))


def test_ch17_mvdr_null():
    m = re.search(r"MVDR ([-\d.]+) dB; output SINR ([-\d.]+) dB vs ([-\d.]+) dB", run("ch17_beamforming.py"))
    assert float(m.group(1)) < -40 and float(m.group(3)) > float(m.group(2))


def test_ch19_gantry_height():
    zs = [float(z) for x, y, z in re.findall(r"x\s+([-+\d.]+)\s+y\s+([\d.]+)\s+z\s+([\d.]+)", run("ch19_4d_point_cloud.py"))
          if float(y) > 28]
    assert zs and all(5.0 < z < 6.2 for z in zs)


def test_ch20_resolution():
    assert "range res c/2B  60.0 cm" in run("ch20_frequency_bands.py")
