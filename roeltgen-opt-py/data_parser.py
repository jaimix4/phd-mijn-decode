# script to get a function Li(ne, Te) out of interpolated ADAS data
# this the equivalent of Li in eq. 1 of Roeltgen's paper

import numpy as np
from scipy.interpolate import RectBivariateSpline


def load_adas_plt_h(filepath):
    """
    Parses a standard ADAS ADF11 (PLT) file for Hydrogen.
    Returns an interpolation function: f(log10_ne, log10_te)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse Grid Sizes (usually the first line)
    # Example: '  9  8' (9 Te points, 8 Ne points)
    header = lines[0].split()
    n_ne = int(header[1])
    n_te = int(header[2])

    # Collect all numbers, skipping metadata
    raw_data = []
    for line in lines[1:]:
        if 'RADIATED POWERS' in line or 'OPEN-ADAS' in line:
            break
        # Clean comments and handle negative numbers properly
        clean_line = line.split('/')[0].replace('-', ' -')
        parts = clean_line.split()
        for p in parts:
            try:
                raw_data.append(float(p.replace('D', 'E')))
            except ValueError:
                continue

    # Slicing ne first, then te
    log10_ne = np.array(raw_data[:n_ne])
    log10_te = np.array(raw_data[n_ne : n_ne + n_te])

    # Extract Matrix: shape (n_ne, n_te)
    data_start = n_ne + n_te
    matrix_vals = np.array(raw_data[data_start : data_start + (n_ne * n_te)])
    lz_matrix = matrix_vals.reshape((n_te, n_ne))
    
    # Create Interpolator: f(log10_ne, log10_te_ev)
    interp = RectBivariateSpline(log10_te, log10_ne, lz_matrix) 
    
    return interp


def get_lz_si(ne, te_ev, interp):
    """
    ne: density in m^-3
    te_ev: temperature in eV
    returns: Lz in W * m^3
    """
    # 1. Evaluate log10(W*cm^3)
    # ADAS ne is usually in cm^-3 (10^13 to 10^20 is typical)
    # If your input ne is m^-3, convert to cm^-3 for the lookup
    ne_cm3 = ne * 1e-6
    log_lz = interp(np.log10(te_ev), np.log10(ne_cm3))[0,0]
    
    # 2. Convert result to W*m^3
    # 10**log_lz gives W*cm^3. Multiply by 1e-6 to get W*m^3
    return (10**log_lz) * 1e-6