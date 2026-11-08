"""radar_sim: a small, readable FMCW / TDM-MIMO radar simulator for learning.

Companion code for the Kymopsis "Radar Basics" series (https://kymopsis.com/blog/).
Textbook point-target model; not a model of any specific product.
"""
from .fmcw import (C, Radar, Target, cube, db, range_fft, doppler_fft, rd_map, angle_spectrum,
                   capon, music, ca_cfar_1d, LAYOUTS, make)

__version__ = "0.1.0"
