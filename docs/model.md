# Signal model

For transmitter t, receiver k, chirp m and ADC sample n, a point target at range R(t), radial velocity v,
azimuth θ and elevation φ produces

    s[m, k, n] = a · exp( j·2π·fb·n/fs  +  j·4π·R(t_m)/λ  +  j·2π·(dx·sinθ·cosφ + dz·sinφ) )

- fb = 2·S·R/c is the beat frequency (S = B/Tc, the chirp slope).
- 4πR/λ is the round-trip carrier phase; its change from chirp to chirp is the Doppler phase.
- (dx, dz) is the position of the virtual element (transmitter position + receiver position) in wavelengths.
- R(t_m) = R0 + v·t_m with t_m the start time of the chirp, so TDM-MIMO motion errors appear naturally.

Processing in `rd_map`: Hann window + range FFT along n, Hann window + Doppler FFT along m (fft-shifted),
then a non-coherent sum over virtual channels for display. The complex range-Doppler cube is returned too,
so angle processing can take a snapshot across channels at any (range, velocity) cell.
