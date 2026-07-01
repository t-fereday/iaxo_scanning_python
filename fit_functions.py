"""
fit_functions.py — Pure-numpy fitting models for surf_align, surf_center, and hist1dfit.

Ported from IDL fit_2d_align.pro, fit_2d_plane.pro, fit_king.pro, and
fit_cylinder.pro.

fit_2d_align:
  Used by surf_align (via scipy.optimize.least_squares) to model the
  misalignment of the shell and the laser scanner apparatus.  The coordinate
  system: x-axis = azimuthal slope (theta_Shell), y-axis = vertical slope
  (z_Shell), z-axis = radial (r_Shell).  Only method=1 is implemented
  (method=0 was dead code in the original IDL).

fit_2d_plane:
  Used by surf_plane_2d.  Simple planar model: c0 + cx*x + cy*y.

fit_king:
  King profile used in hist1dfit: A0 / (1 + ((x - x0) / r)^2)^alpha.

fit_cylinder:
  Used by surf_center (via scipy.optimize.curve_fit).  Models the radial
  deviation of a cylindrical shell with centre offset (h0, alpha0).
  Replaces the IDL COMMON block cradius by taking r0 as an explicit argument.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL fit_2d_align.pro, fit_2d_plane.pro, fit_king.pro,
    fit_cylinder.pro
"""
import numpy as np


def fit_2d_align(atheta, az, par):
    """
    Misalignment model for surf_align (method=1 branch).

    Parameters: [r0, D0, etax, etay, cz, c0, h0, alpha0]
    Returns the predicted surface offset at each (atheta, az) point.
    """
    etax   = par[2]
    etay   = par[3]
    cz     = par[4]
    c0     = par[5]
    h0     = par[6]
    alpha0 = par[7]

    x0 = h0 * np.cos(alpha0)
    y0 = h0 * np.sin(alpha0)

    return (etay * az * np.sin(atheta)
            + etax * az * np.cos(atheta)
            + x0 * np.cos(atheta)
            + y0 * np.sin(atheta)
            + cz * az + c0)


def fit_2d_plane(x1, x2, par):
    """Planar surface model: c0 + cx1*x1 + cx2*x2."""
    return par[0] + par[1] * x1 + par[2] * x2


def fit_king(data, par):
    """
    King profile: A0 / (1 + ((data - x2) / r2)^2)^alpha2.

    par = [A0, x2, r2, alpha2]
    """
    A0     = par[0]
    x2     = par[1]
    r2     = par[2]
    alpha2 = par[3]
    return A0 / (1.0 + ((data - x2) / r2) ** 2) ** alpha2


def fit_cylinder(vtheta, a0, h0, alpha0, r0):
    """
    Cylindrical offset model for surf_center.

    Returns predicted radial deviation given shell center offset (h0, alpha0).
    r0 is the nominal radius (fixed, passed as extra arg).
    """
    cosval  = np.cos(vtheta - alpha0)
    sinval  = np.sin(vtheta - alpha0)
    sqrtval = np.sqrt(np.maximum(a0 ** 2 - h0 ** 2 * sinval ** 2, 0.0))
    return h0 * cosval + sqrtval - r0
