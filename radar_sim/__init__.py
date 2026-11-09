"""radar_sim: a small, readable FMCW / TDM-MIMO radar simulator for learning.

Companion code for the Kymopsis "Radar Principles" series (https://kymopsis.com/blog/).
Textbook point-target model; not a model of any specific product.
"""
from .fmcw import (C, Radar, Target, cube, db, range_fft, doppler_fft, rd_map, angle_spectrum,
                   capon, music, ca_cfar_1d, LAYOUTS, make)
from .pipeline import (ca_cfar_2d, peaks_2d, doppler_dft_at, angle_fft_all, angle_dft_at, angle_split_phase,
                       op_counts, tdm_phase_comp, beam_monopulse, KalmanCV, Tracker)

__version__ = "0.2.2"
