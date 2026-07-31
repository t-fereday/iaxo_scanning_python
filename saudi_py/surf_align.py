"""
surf_align.py — Align laser-scanner surface data for shell misalignment.

Ported from IDL surf_align.pro.

Corrects for misalignments in the position of the shell and in the laser
scanner apparatus by fitting a parametric misalignment model (fit_2d_align,
method=1) to the gridded surface.  The 8 fit parameters are:
  [r0, D0, etax, etay, cz, c0, h0, alpha0]
where D0 is fixed and (when realign=True) cz is also fixed.

The procedure concentrates on shell misalignment (h0, alpha0) — the scanner
misalignment terms (etax, etay) can be highly correlated and often yield
unphysical values.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, July 2002
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_align.pro; mpfit2dfun replaced with
    scipy.optimize.least_squares (trf method)
"""
import numpy as np
from scipy.optimize import least_squares
from .fit_functions import fit_2d_align


def surf_align(theta, zaxis, dr, r0=80.0, D0=600.0,
               ia=None, ib=None, ja=None, jb=None,
               etax=0.0, etay=0.0, cz=0.0, c0=0.0, h0=0.0, alpha0=0.0,
               realign=False, quiet=False):
    """
    Align laser-scanner surface data by fitting a misalignment model.

    Equivalent to IDL surf_align (uses mpfit2dfun with fit_2d_align, method=1).

    Parameters
    ----------
    theta  : (ntheta,) array-like   azimuthal angles [rad]
    zaxis  : (nz,)     array-like   axial coordinates [mm]
    dr     : (ntheta, nz) array-like  surface heights [mm]
    r0     : float   shell radius [mm]
    D0     : float   PSD distance [mm]
    ia, ib, ja, jb : int, optional   inclusive fit sub-range indices
    etax, etay, cz, c0, h0, alpha0 : float   initial parameter guesses
    realign : bool   if True, fix cz=0 (plane-z not fitted)
    quiet   : bool   suppress convergence output

    Returns
    -------
    dr_aligned : (ntheta, nz) ndarray
    fit_par    : (8,) ndarray  [r0, D0, etax, etay, cz, c0, h0, alpha0]
    """
    theta = np.asarray(theta, dtype=float)
    zaxis = np.asarray(zaxis, dtype=float)
    dr    = np.asarray(dr,    dtype=float)

    ilen, jlen = dr.shape

    ia = 0      if ia is None else int(ia)
    ib = ilen-1 if ib is None else int(ib)
    ja = 0      if ja is None else int(ja)
    jb = jlen-1 if jb is None else int(jb)

    ia = max(ia, 0); ib = min(ib, ilen-1)
    ja = max(ja, 0); jb = min(jb, jlen-1)

    # Build 2-D grids centred on the means (as IDL does)
    theta_mean = np.mean(theta)
    z_mean     = np.mean(zaxis)

    atheta_full = np.outer(theta - theta_mean, np.ones(jlen))
    az_full     = np.outer(np.ones(ilen), zaxis - z_mean)

    atheta_sub = atheta_full[ia:ib+1, ja:jb+1]
    az_sub     = az_full[ia:ib+1, ja:jb+1]
    dr_sub     = dr[ia:ib+1, ja:jb+1]

    # Parameter order: [r0, D0, etax, etay, cz, c0, h0, alpha0]
    # D0 and (when realign) cz are fixed — exclude them from optimisation,
    # pass as constants so scipy never sees lb == ub.
    fixed_indices  = [1]               # D0 always fixed
    fixed_values   = [D0]
    if realign:
        fixed_indices.append(4)        # cz fixed when realigning
        fixed_values.append(0.0)

    free_indices = [i for i in range(8) if i not in fixed_indices]
    full_start   = np.array([r0, D0, etax, etay, cz, c0, h0, alpha0], dtype=float)
    start_free   = full_start[free_indices]

    def _build_full(free_par):
        p = full_start.copy()
        for fi, fv in zip(fixed_indices, fixed_values):
            p[fi] = fv
        for idx, val in zip(free_indices, free_par):
            p[idx] = val
        return p

    def residuals(free_par):
        par   = _build_full(free_par)
        model = fit_2d_align(atheta_sub, az_sub, par)
        return (dr_sub - model).ravel()

    result  = least_squares(residuals, start_free, method='trf')
    fit_par = _build_full(result.x)

    dr_fit  = fit_2d_align(atheta_full, az_full, fit_par)
    return dr - dr_fit, fit_par
