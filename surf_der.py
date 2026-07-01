"""
surf_der.py — First partial derivatives of a 2-D surface.

Ported from IDL surf_der.pro.

Generates the first partial derivatives (slopes) of surface data using the
IDL DERIV function (3-point Lagrangian differencing), approximated here
with numpy.gradient (also 3-point at interior points, 1-sided at edges).

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_der.pro; IDL DERIV replaced with numpy.gradient
"""
import numpy as np


def surf_der(vx, vy, az):
    """
    Compute first partial derivatives of a 2-D surface.

    Equivalent to IDL surf_der (uses IDL DERIV = 3-point Lagrangian,
    approximated here with numpy.gradient).

    Parameters
    ----------
    vx : (ntheta,) array-like   azimuthal coordinates
    vy : (nz,)     array-like   axial coordinates
    az : (ntheta, nz) array-like  surface heights

    Returns
    -------
    dzdx : (ntheta, nz) ndarray  ∂z/∂theta
    dzdy : (ntheta, nz) ndarray  ∂z/∂z_axis
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    az = np.asarray(az, dtype=float)

    imax, jmax = az.shape

    dzdx = np.zeros_like(az)
    dzdy = np.zeros_like(az)

    if imax >= 3:
        for j in range(jmax):
            dzdx[:, j] = np.gradient(az[:, j], vx)

    for i in range(imax):
        dzdy[i, :] = np.gradient(az[i, :], vy)

    return dzdx, dzdy
