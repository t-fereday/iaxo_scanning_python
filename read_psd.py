"""
read_psd.py — Read formatted laser-scanner PSD data files.

Ported from IDL read_PSD.pro.

The laser-scanner data files are text files containing five columns per row:
  col 0  PSDdata[0]  Voltage for horizontal PSD position [V]  (2 mm/V)
  col 1  PSDdata[1]  Voltage for vertical PSD position [V]    (2 mm/V)
  col 2  PSDdata[2]  Laser signal intensity [V]
  col 3  PSDdata[3]  Angular position of the scanner PSD [degrees * 1000]
                       THETA = PSDdata[3] * pi / 180 / 1000
  col 4  PSDdata[4]  Vertical stage position [micrometers]
                       Z = PSDdata[4] / 1000  [mm]

Reference:
  "Fast optical metrology of the hard x-ray optics for the HEFT telescope",
  Mario Jimenez-Garate et al., X-Ray Optics, Instruments, and Missions,
  SPIE Proceedings Vol. 3444, 1998.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL read_psd.pro
"""
import os
import numpy as np


def read_psd(infile, noprint=False):
    """
    Read a formatted laser scanner PSD data file.

    File format: 5 whitespace-delimited columns per row:
      col 0 – PSDx voltage [V]
      col 1 – PSDy voltage [V]
      col 2 – laser signal intensity [V]
      col 3 – azimuthal angle [degrees * 1000]
      col 4 – axial position [micrometers]

    Returns
    -------
    PSDx, PSDy, signal, theta, zaxis : numpy.ndarray or None
        None for all arrays when error != 0.
    error : int
        0 on success, non-zero on failure.
    filedate : float
        File mtime (seconds since epoch), or 0 on failure.
    """
    error = 0
    filedate = 0.0

    if not os.path.isfile(infile):
        if not noprint:
            print(f"read_psd: file not found — {infile}")
        return None, None, None, None, None, 1, filedate

    try:
        with open(infile, 'r') as f:
            raw = f.read()
        # Replace commas with spaces so the file can be read as a flat
        # whitespace-delimited stream — mirrors IDL readf behaviour which
        # reads N values regardless of line boundaries.
        values = np.fromstring(raw.replace(',', ' '), dtype=float, sep=' ')
        if values.size % 5 != 0:
            values = values[:values.size - (values.size % 5)]
        data = values.reshape(-1, 5)
    except Exception as exc:
        if not noprint:
            print(f"read_psd: read error — {infile}: {exc}")
        return None, None, None, None, None, 1, filedate

    if data.shape[0] < 4 or data.shape[1] < 5:
        print("read_psd: not enough scan data to analyze")
        return None, None, None, None, None, 1, filedate

    filedate = os.path.getmtime(infile)

    if not noprint:
        print(f"data read from {infile}, size = {data.shape[0]}")

    PSDx   =  data[:, 0]                       # raw voltage [V]
    PSDy   =  data[:, 1]
    signal =  data[:, 2]
    theta  = -data[:, 3] / 1000.0 / 180.0 * np.pi   # → radians
    zaxis  =  data[:, 4] / 1000.0                    # μm → mm

    return PSDx, PSDy, signal, theta, zaxis, error, filedate
