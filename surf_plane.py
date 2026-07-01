"""
surf_plane.py — Subtract a best-fit plane from a 2-D surface array.

Ported from IDL surf_plane.pro and surf_plane_2d.pro.

surf_plane uses uniform integer index grids (equivalent to IDL REGRESS with
integer x1/x2 arrays).

surf_plane_2d uses physical coordinate grids (vx, vy) and is equivalent to
IDL surf_plane_2d which used mpfit2dfun with fit_2d_plane.  Since the
planar model is linear, numpy.linalg.lstsq gives the exact same result
without an iterative optimizer.

The fit is performed over the sub-region [ia:ib, ja:jb] (inclusive) and
the fitted plane is subtracted from the entire array.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_plane.pro and surf_plane_2d.pro;
    IDL REGRESS / mpfit2dfun replaced with numpy.linalg.lstsq
"""
import numpy as np


def surf_plane(z, ia=None, ib=None, ja=None, jb=None):
    """
    Subtract a best-fit plane from a 2-D surface array using uniform index grids.

    Equivalent to IDL surf_plane (uses REGRESS with integer x1/x2 grids).

    Parameters
    ----------
    z : (ntheta, nz) numpy.ndarray
    ia, ib : int, optional
        First/last theta indices used in the fit (inclusive). Default: full range.
    ja, jb : int, optional
        First/last z indices used in the fit (inclusive). Default: full range.

    Returns
    -------
    numpy.ndarray
        z with the fitted plane subtracted (same shape as input).
    """
    z = np.array(z, dtype=float)
    mx, my = z.shape

    ia = 0    if ia is None else int(ia)
    ib = mx-1 if ib is None else int(ib)
    ja = 0    if ja is None else int(ja)
    jb = my-1 if jb is None else int(jb)

    # Integer coordinate grids over the full array
    x1_full = np.tile(np.arange(mx)[:, np.newaxis], (1, my)).astype(float)
    x2_full = np.tile(np.arange(my)[np.newaxis, :], (mx, 1)).astype(float)

    # Fit sub-region
    sub_x1 = x1_full[ia:ib+1, ja:jb+1].ravel()
    sub_x2 = x2_full[ia:ib+1, ja:jb+1].ravel()
    sub_z  = z[ia:ib+1, ja:jb+1].ravel()

    A = np.column_stack([np.ones_like(sub_x1), sub_x1, sub_x2])
    coeffs, _, _, _ = np.linalg.lstsq(A, sub_z, rcond=None)

    z_fit = coeffs[0] + coeffs[1] * x1_full + coeffs[2] * x2_full
    return z - z_fit


def surf_plane_2d(az, ia=None, ib=None, ja=None, jb=None, vx=None, vy=None):
    """
    Subtract a best-fit plane from a 2-D surface using physical coordinates.

    Equivalent to IDL surf_plane_2d (uses mpfit2dfun with fit_2d_plane).
    Since the model is linear, this is solved exactly with lstsq.

    Parameters
    ----------
    az : (ntheta, nz) numpy.ndarray
    ia, ib, ja, jb : int, optional  (inclusive limits, default: full range)
    vx : (ntheta,) array-like, optional  azimuthal coordinates
    vy : (nz,)     array-like, optional  axial coordinates

    Returns
    -------
    numpy.ndarray
        az with the fitted plane subtracted.
    """
    az = np.array(az, dtype=float)
    mx, my = az.shape

    ia = 0    if ia is None else int(ia)
    ib = mx-1 if ib is None else int(ib)
    ja = 0    if ja is None else int(ja)
    jb = my-1 if jb is None else int(jb)

    if vx is None:
        vx = np.arange(mx, dtype=float)
    if vy is None:
        vy = np.arange(my, dtype=float)

    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    # 2-D coordinate grids over the full array
    ax = np.outer(vx, np.ones(my))
    ay = np.outer(np.ones(mx), vy)

    # Fit sub-region
    sub_ax = ax[ia:ib+1, ja:jb+1].ravel()
    sub_ay = ay[ia:ib+1, ja:jb+1].ravel()
    sub_az = az[ia:ib+1, ja:jb+1].ravel()

    A = np.column_stack([np.ones_like(sub_ax), sub_ax, sub_ay])
    par, _, _, _ = np.linalg.lstsq(A, sub_az, rcond=None)

    z_fit = par[0] + par[1] * ax + par[2] * ay
    return az - z_fit
