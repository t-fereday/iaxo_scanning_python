"""
surf_gen.py — Generate a surface height map from laser-scanner slope data.

Ported from IDL surf_gen.pro.

From the laser-scanner data the slope in both the axial and azimuthal
directions is inferred.  The axial scan for each azimuthal setting will have
a relative radial offset and a slope offset in the axial direction due to
small changes in the vertical position of the scanner PSD.  These offsets are
determined by fitting the following linear system:

  g(i,j+1) = g(i,j) + dgdz(i,j)*[z(j+1)-z(j)]            (1)  axial integration
  f(i,j)   = g(i,j) - [a(i)*z(j) + b(i)]                  (2)  offset correction
  f(i+1,j) = g(i,j) + dgdtheta(i,j)*[theta(i+1)-theta(i)] (3)  azimuthal stitching

Equating (2) and (3) gives a linear equation for the per-scan slope offsets
c(i) and radial offsets d(i), which are accumulated as cumulative sums to
obtain a(i) and b(i).  An overall plane is then removed from the result.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_gen.pro
"""
import numpy as np
from .surf_plane import surf_plane


def surf_gen(vx, vy, dzdx, dzdy):
    """
    Generate a surface height map by integrating azimuthal and axial slope data.

    Equivalent to IDL surf_gen.pro.  The algorithm:
      1. Integrate azimuthal slopes to get a reference row (azg[:,0]).
      2. Integrate axial slopes row-by-row to fill azg[:,1:].
      3. Fit residual offsets (slope + bias) between adjacent azimuthal scans
         using a linear poly-fit.
      4. Accumulate the offsets (va, vb) and subtract them from azg.
      5. Remove an overall plane from the result.

    Parameters
    ----------
    vx : (ntheta,) array-like   azimuthal coordinates [rad]
    vy : (nz,)     array-like   axial coordinates [mm]
    dzdx : (ntheta, nz) array-like  azimuthal surface slopes
    dzdy : (ntheta, nz) array-like  axial surface slopes

    Returns
    -------
    gdr0   : (ntheta, nz) ndarray   reconstructed height residuals [mm]
    va     : (ntheta,)   ndarray   cumulative axial slope offsets
    vb     : (ntheta,)   ndarray   cumulative radial offsets
    vsiga  : (ntheta,)   ndarray   uncertainty in va
    vsigb  : (ntheta,)   ndarray   uncertainty in vb
    drx    : (ntheta, nz) ndarray  raw azimuthal height differences
    dry    : (ntheta, nz) ndarray  axially-integrated heights (azg)
    kspacer : None  (placeholder for optional spacer mask)
    """
    vx   = np.asarray(vx,   dtype=float)
    vy   = np.asarray(vy,   dtype=float)
    dzdx = np.asarray(dzdx, dtype=float)
    dzdy = np.asarray(dzdy, dtype=float)

    ilen = len(vx)
    jlen = len(vy)

    azg = np.zeros((ilen, jlen))
    azh = np.zeros((ilen, jlen))
    va  = np.zeros(ilen)
    vb  = np.zeros(ilen)
    vc  = np.zeros(ilen)
    vd  = np.zeros(ilen)
    vsigc = np.zeros(ilen)
    vsigd = np.zeros(ilen)

    # Integrate azimuthal slope at j=0
    azg[0, 0] = 0.0
    for i in range(1, ilen):
        azg[i, 0] = azg[i-1, 0] + (dzdx[i, 0] + dzdx[i-1, 0]) / 2.0 * (vx[i] - vx[i-1])

    # Integrate axial slope for each j
    for j in range(1, jlen):
        azg[:, j] = azg[:, j-1] + (dzdy[:, j] + dzdy[:, j-1]) / 2.0 * (vy[j] - vy[j-1])

    dry = azg.copy()

    # Azimuthal height differences (used as drx)
    azh[1:ilen, :] = (azg[1:ilen, :] - azg[0:ilen-1, :]
                      - (dzdx[1:ilen, :] + dzdx[0:ilen-1, :]) / 2.0
                      * np.outer(vx[1:ilen] - vx[0:ilen-1], np.ones(jlen)))

    drx = azh.copy()
    drx[1:ilen, :] = drx[1:ilen, :] - (azg[1:ilen, :] - azg[0:ilen-1, :])

    # Fit range: inner 80 % of axial range
    ja = int(jlen * 0.1)
    jb = int(jlen * 0.9)

    for i in range(1, ilen):
        coeffs = np.polyfit(vy[ja:jb+1], azh[i, ja:jb+1], 1)
        # polyfit returns [c1, c0]; IDL poly_fit returns [c0, c1]
        vd[i] = coeffs[1]   # constant term
        vc[i] = coeffs[0]   # slope term
        cov = np.cov(vy[ja:jb+1], azh[i, ja:jb+1])
        # Rough sigma from residuals
        yfit = np.polyval(coeffs, vy[ja:jb+1])
        residuals = azh[i, ja:jb+1] - yfit
        n = jb - ja + 1
        if n > 2:
            sigma_y = np.std(residuals) / np.sqrt(n - 2)
            # Rough covariance-based sigma on slope and intercept
            sx = np.std(vy[ja:jb+1])
            vsigc[i] = sigma_y / sx if sx > 0 else 0.0
            vsigd[i] = sigma_y * np.sqrt(np.mean(vy[ja:jb+1]**2)) / sx if sx > 0 else 0.0

    va    = np.cumsum(vc)
    vb    = np.cumsum(vd)
    vsiga = np.sqrt(np.cumsum(vsigc ** 2))
    vsigb = np.sqrt(np.cumsum(vsigd ** 2))

    azf = np.zeros((ilen, jlen))
    for i in range(ilen):
        azf[i, :] = azg[i, :] - (va[i] * vy + vb[i])

    gdr0 = surf_plane(azf)

    return gdr0, va, vb, vsiga, vsigb, drx, dry, None
