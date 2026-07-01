"""
surf_calibrate.py — Load and average laser-scanner calibration PSD data.

Ported from IDL surf_calibrate.pro.

Calibration data is recorded for an optical lambda/5 test flat and averaged
over all calibration scans found in file_dir.  The primary correction
accounts for the misalignment of the upward laser trajectory with respect
to the vertical stage, which produces a linear shift in both PSDx and PSDy
as a function of vertical stage position.  After averaging, the calibration
vectors are mean-subtracted over the specified zcut range.

The calibration files are text files with the same 5-column format as the
regular scan files (see read_psd.py).

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, May 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_calibrate.pro; IDL interpol(/lsquadratic)
    replaced with scipy.interpolate.interp1d(kind='quadratic')
"""
import glob
import os
import numpy as np
from scipy.interpolate import interp1d
from .read_psd import read_psd


def surf_calibrate(zmin=None, zmax=None, zstep=1.0,
                   file_base='cal_01172024',
                   file_dir=r'C:\IAXOscan\calibration\\',
                   file_num=None,
                   fitdeg=1,
                   zcut=None,
                   newscanner=True,
                   noplot=True,
                   charsize=1,
                   position=12):
    """
    Load and average laser calibration PSD data.

    Equivalent to IDL surf_calibrate.

    Parameters
    ----------
    zmin, zmax : float
        Axial range for the calibration grid [mm].
    zstep : float
        Axial grid step [mm].
    file_base : str
        Base name of the calibration file set (e.g. ``'cal_01172024'``).
    file_dir : str
        Directory containing calibration files.
    file_num : list of str, optional
        Explicit list of scan numbers (e.g. ``['d1','d2','d3']``).
        If None, all ``<file_base>*.txt.*`` files are used.
    fitdeg : int
        Degree of polynomial fit used when plotting (1 = linear).
    zcut : [zmin_cut, zmax_cut], optional
        Axial range used for mean-subtraction; defaults to [zmin+10, zmax-10].
    newscanner : bool
        Apply z-offset of 150 mm for the new scanner.
    noplot : bool
        Suppress calibration plots (default True).

    Returns
    -------
    PSDx0, PSDy0, signal0, theta0, zaxis0 : numpy.ndarray
        Averaged calibration vectors on the uniform axial grid.
    """
    zoffset = 150.0 if newscanner else 0.0

    if zmin is None:
        zmin = 53.0
    if zmax is None:
        zmax = 240.0

    if zcut is None:
        zcut = [zmin + 10.0, zmax - 10.0]
    zcut[0] = max(min(zcut[0], zmax - 3.0), zmin)
    zcut[1] = min(max(zcut[1], zmin + 3.0), zmax)

    isize  = int((zmax - zmin) / zstep) + 1
    zaxis0 = np.arange(isize, dtype=float) * zstep + zmin
    PSDx0  = np.zeros(isize)
    PSDy0  = np.zeros(isize)
    signal0 = np.zeros(isize)
    theta0  = np.zeros(isize)

    if file_num is not None:
        cal_files = [os.path.join(file_dir, f"{file_base}.txt.{fn}") for fn in file_num]
    else:
        pattern  = os.path.join(file_dir, f"{file_base}*.txt.*")
        cal_files = sorted(glob.glob(pattern))

    n_files = len(cal_files)
    if n_files == 0:
        print(f"surf_calibrate: WARNING no calibration files found matching {pattern!r}")
        return PSDx0, PSDy0, signal0, theta0, zaxis0

    for file_name in cal_files:
        vPSDx, vPSDy, vsignal, vtheta, vzaxis1, error, _ = read_psd(file_name, noprint=True)
        if error != 0 or vPSDx is None:
            continue

        vzaxis = vzaxis1 + zoffset

        def _interp(y, x, x0):
            # Clamp x0 to data range to avoid extrapolation errors
            x0_clipped = np.clip(x0, x.min(), x.max())
            f = interp1d(x, y, kind='quadratic', bounds_error=False,
                         fill_value=(y[0], y[-1]))
            return f(x0_clipped)

        PSDx0   += _interp(vPSDx,   vzaxis, zaxis0)
        PSDy0   += _interp(vPSDy,   vzaxis, zaxis0)
        signal0 += _interp(vsignal, vzaxis, zaxis0)
        theta0  += _interp(vtheta,  vzaxis, zaxis0)

    PSDx0   /= n_files
    PSDy0   /= n_files
    signal0 /= n_files
    theta0  /= n_files

    # Mean-subtract over the zcut range
    ia = 0
    while ia < isize - 1 and zaxis0[ia] < zcut[0]:
        ia += 1
    ib = isize - 1
    while ib > 0 and zaxis0[ib] > zcut[1]:
        ib -= 1

    PSDx0 -= np.mean(PSDx0[ia:ib+1])
    PSDy0 -= np.mean(PSDy0[ia:ib+1])

    if not noplot:
        _plot_calibration(zaxis0, PSDx0, PSDy0, signal0, ia, ib, fitdeg)

    return PSDx0, PSDy0, signal0, theta0, zaxis0


def _plot_calibration(zaxis0, PSDx0, PSDy0, signal0, ia, ib, fitdeg):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(zaxis0[ia:ib+1], PSDx0[ia:ib+1])
    axes[0, 0].set_xlabel('zaxis [mm]'); axes[0, 0].set_ylabel('PSDx [mm]')
    coeffs = np.polyfit(zaxis0[ia:ib+1], PSDx0[ia:ib+1], fitdeg)
    axes[0, 0].plot(zaxis0[ia:ib+1], np.polyval(coeffs, zaxis0[ia:ib+1]), 'r--')

    axes[0, 1].plot(zaxis0[ia:ib+1], PSDy0[ia:ib+1])
    axes[0, 1].set_xlabel('zaxis [mm]'); axes[0, 1].set_ylabel('PSDy [mm]')
    coeffs = np.polyfit(zaxis0[ia:ib+1], PSDy0[ia:ib+1], fitdeg)
    axes[0, 1].plot(zaxis0[ia:ib+1], np.polyval(coeffs, zaxis0[ia:ib+1]), 'r--')

    axes[1, 0].plot(zaxis0[ia:ib+1], signal0[ia:ib+1])
    axes[1, 0].set_xlabel('zaxis [mm]'); axes[1, 0].set_ylabel('signal [V]')

    axes[1, 1].axis('off')

    fig.tight_layout()
    plt.show()
