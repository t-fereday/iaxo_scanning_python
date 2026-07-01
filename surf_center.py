"""
surf_center.py — Centre a laser-scanned mirror surface by fitting a cylindrical offset.

Ported from IDL surf_center.pro.

Fits the mean azimuthal profile to a cylindrical offset model (fit_cylinder)
to determine the shell centre offset (h0, alpha0) relative to the scanner
axis.  The fit parameters are:
  a0     : actual radius of the shell [mm]
  h0     : radial centre offset of the shell [mm]
  alpha0 : angle for the centre offset [rad]

The surface heights are then corrected to remove the centre offset, and
the azimuthal angles are updated accordingly.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_center.pro; IDL curvefit replaced with
    scipy.optimize.curve_fit; IDL COMMON block cradius replaced by
    explicit r0 argument to fit_cylinder
"""
import numpy as np
from scipy.optimize import curve_fit
from .fit_functions import fit_cylinder


def surf_center(theta, z, dr, r0=80.0, ia=None, ib=None, ja=None, jb=None):
    """
    Centre a surface generated from laser-scanner data.

    Fits the mean azimuthal profile to a cylindrical offset model to find
    the shell centre offset (h0, alpha0) with respect to the scanner axis,
    then corrects the surface heights accordingly.

    Equivalent to IDL surf_center.

    Parameters
    ----------
    theta : (ntheta,) array-like   azimuthal angles [rad]
    z     : (nz,)     array-like   axial coordinates [mm]
    dr    : (ntheta, nz) array-like  surface heights [mm]
    r0    : float   nominal shell radius [mm]
    ia, ib, ja, jb : int, optional  inclusive fit sub-range indices

    Returns
    -------
    adrnew   : (ntheta, nz) ndarray  corrected surface heights
    thetanew : (ntheta,)   ndarray  corrected azimuthal angles
    par      : [a0, h0, alpha0]
    """
    theta = np.asarray(theta, dtype=float)
    z     = np.asarray(z,     dtype=float)
    dr    = np.asarray(dr,    dtype=float)

    imax, jmax = dr.shape

    ia = 0     if ia is None else int(ia)
    ib = imax-1 if ib is None else int(ib)
    ja = 0     if ja is None else int(ja)
    jb = jmax-1 if jb is None else int(jb)

    ia = max(ia, 0); ib = min(ib, imax-1)
    ja = max(ja, 0); jb = min(jb, jmax-1)

    # Flatten theta and dr sub-region into 1-D vectors
    atheta_sub = np.tile(theta[ia:ib+1, np.newaxis], (1, jb-ja+1)).ravel()
    vr         = dr[ia:ib+1, ja:jb+1].ravel()

    # Initial guesses
    a0     = float(r0)
    h0_0   = float(dr[int((ib+ia)/2), int((jb+ja)/2)])
    alpha0_0 = float(np.mean(atheta_sub))

    def _model(vtheta, a0_, h0_, alpha0_):
        return fit_cylinder(vtheta, a0_, h0_, alpha0_, r0)

    try:
        popt, _ = curve_fit(_model, atheta_sub, vr,
                             p0=[a0, h0_0, alpha0_0],
                             maxfev=10000)
        a0_fit, h0_fit, alpha0_fit = popt
    except Exception:
        # Fall back to initial guesses if fit fails
        a0_fit, h0_fit, alpha0_fit = a0, 0.0, float(np.mean(theta))

    # Build full 2-D theta grid
    atheta = np.tile(theta[:, np.newaxis], (1, jmax))

    adrnew = (np.sqrt(np.maximum(
                  ((r0 + dr) * np.cos(atheta) - h0_fit * np.cos(alpha0_fit)) ** 2
                + ((r0 + dr) * np.sin(atheta) - h0_fit * np.sin(alpha0_fit)) ** 2,
                  0.0))
              - a0_fit)

    thetanew = np.arctan(np.tan(theta) + h0_fit / r0 * np.sin(theta - alpha0_fit))

    return adrnew, thetanew, [a0_fit, h0_fit, alpha0_fit]
