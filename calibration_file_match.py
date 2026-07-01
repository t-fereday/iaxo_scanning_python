"""
calibration_file_match.py — Find the calibration file set closest in time to a scan file.

Ported from IDL calibration_file_match.pro.

Searches dir_calib for all *.txt.1 files, compares their modification times
to filedate (an os.path.getmtime value), and returns the base name of the
calibration set whose mtime is closest.

Modification History:
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL calibration_file_match.pro; IDL file_info.mtime
    replaced with os.path.getmtime; IDL file_search replaced with glob.glob
"""
import glob
import os


def calibration_file_match(filedate, dir_calib=r'C:\NuSTAR\Scan\calibration\\'):
    """
    Find the calibration file set whose date is closest to *filedate*.

    Parameters
    ----------
    filedate : float
        Reference file mtime in seconds since epoch (from os.path.getmtime).
    dir_calib : str
        Directory containing calibration scan files named like
        ``<base>.txt.1``, ``<base>.txt.2``, etc.

    Returns
    -------
    str
        The base name of the closest calibration set (without extension),
        e.g. ``'cal_01172024'``.
    """
    pattern = os.path.join(dir_calib, '*.txt.1')
    cal_files = glob.glob(pattern)

    if not cal_files:
        raise FileNotFoundError(
            f"calibration_file_match: no calibration files found in {dir_calib!r}"
        )

    mtimes = [os.path.getmtime(f) for f in cal_files]
    diffs = [abs(t - filedate) for t in mtimes]
    best = cal_files[diffs.index(min(diffs))]

    # Strip .txt.1 → base name
    basename = os.path.basename(best)
    # Remove last two extensions (.txt and .1)
    base = os.path.splitext(os.path.splitext(basename)[0])[0]
    return base
