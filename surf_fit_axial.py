"""
surf_fit_axial.py — Subtract a polynomial fit from each azimuthal row of surface data.

Ported from IDL surf_fit_axial.pro.

Removes a polynomial of degree FITDEG from each row (azimuthal scan) of the
surface height array DR.  This is used to remove phase (linear, fitdeg=1)
or bow (quadratic, fitdeg=2) contributions from the surface.

The fit coefficients are returned in IDL order (low to high power) for
compatibility with downstream IDL-convention code.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Dec 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_fit_axial.pro; IDL POLY_FIT (low→high coeff
    order) replicated via np.polyfit with [::-1] reversal on output
"""
import numpy as np


def surf_fit_axial(dr, fitdeg=0, zaxis=None, theta=None, fraction=1.0):
    """
    Subtract a polynomial fit of degree *fitdeg* from each azimuthal row of *dr*.

    Equivalent to IDL surf_fit_axial.

    Parameters
    ----------
    dr : (ntheta, nz) array-like
        Surface height array.
    fitdeg : int
        Degree of the polynomial fit along the axial direction.
    zaxis : (nz,) array-like, optional
        Axial coordinates (default: integer indices).
    theta : (ntheta,) array-like, optional
        Azimuthal coordinates (unused in the fit, kept for API parity).
    fraction : float
        Fraction of the axial range to use in the fit (default 1.0 = full range).

    Returns
    -------
    dr_residual : (ntheta, nz) ndarray
        dr with the polynomial fit subtracted.
    fresult : (ntheta, fitdeg+1) ndarray
        Fit coefficients in IDL order (low → high power), shape (ntheta, fitdeg+1).
    """
    dr = np.array(dr, dtype=float)
    ntheta, nzaxis = dr.shape

    if zaxis is None:
        zaxis = np.arange(nzaxis, dtype=float)
    zaxis = np.asarray(zaxis, dtype=float)

    ja = int(nzaxis / 2.0 * (1.0 - fraction))
    jb = int(nzaxis - nzaxis / 2.0 * fraction - 1)

    dr_fit  = np.zeros_like(dr)
    fresult = np.zeros((ntheta, fitdeg + 1))

    for i in range(ntheta):
        # np.polyfit returns coefficients high→low; reverse to IDL low→high
        coeffs_np = np.polyfit(zaxis, dr[i, :], fitdeg)
        yfit = np.polyval(coeffs_np, zaxis)
        dr_fit[i, :] = yfit
        fresult[i, :] = coeffs_np[::-1]   # low→high for output compatibility

    return dr - dr_fit, fresult
