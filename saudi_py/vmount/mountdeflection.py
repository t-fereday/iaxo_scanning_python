"""
mountdeflection.py — Beam deflection required to meet a set of point constraints.

Ported from IDL MountDeflection.pro (Jason Koglin, CAL, Feb 2006).

Computes the deflection curve of an Euler-Bernoulli beam that is forced to pass
through specified deflections ``Yconstraint`` at positions ``Xconstraint``
(here the abscissa is the azimuthal angle theta).  Physically it represents the
point forces ``F_i`` (and, in the moment variant, moments ``M_i``) applied at the
spacer positions to pull the free-standing shell down onto the spacers; between
loads the deflection is a piecewise cubic (the ``w'''' = point load`` solution),
and outside the end constraints it extrapolates linearly at the end slope.

  - Default (``dYconstraint is None``): only slope *continuity* is enforced at
    interior constraints; unknowns are ``[F_0, ..., F_{N-1}, A]`` where A is the
    overall slope rotation (N+1 unknowns).
  - With ``dYconstraint``: N-1 moments are added and both height and slope are
    matched at each constraint; unknowns are ``[F_0..F_{N-1}, M_0..M_{N-2}]``
    (2N-1 unknowns).

The IDL routine assembles ``[Matrix][Svector] = [Bvector]`` and solves by SVD
(SVDC/SVSOL); here we build the same system and solve with numpy.linalg.lstsq
(SVD-based least squares), which reproduces the SVSOL pseudo-inverse behaviour.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2006
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL MountDeflection.pro; IDL SVDC/SVSOL replaced with
    numpy.linalg.lstsq.  Note: IDL indexes Matrix(col, row); this port builds
    M[row, col] directly.
"""
import numpy as np


def mountdeflection(xaxis, xconstraint, yconstraint, dyconstraint=None):
    """Return the beam deflection at ``xaxis`` for the given constraints.

    Equivalent to IDL ``MountDeflection(xaxis, Xconstraint, Yconstraint
    [, dYconstraint])``.

    Parameters
    ----------
    xaxis : array-like
        Positions at which the deflection is evaluated (same units as
        ``xconstraint``).
    xconstraint : array-like
        Positions at which a deflection ``yconstraint`` is imposed.
    yconstraint : array-like
        Imposed deflection amount at each ``xconstraint`` (result units).
    dyconstraint : array-like, optional
        Imposed slopes at the constraint positions.  If omitted, slopes are
        free and only continuity is required.

    Returns
    -------
    ndarray
        Deflections at the ``xaxis`` positions.
    """
    xaxis = np.asarray(xaxis, dtype=float)
    xc = np.asarray(xconstraint, dtype=float)
    yc = np.asarray(yconstraint, dtype=float)
    n = len(xc)

    yaxis = np.zeros(len(xaxis))

    if dyconstraint is not None:
        dyc = np.asarray(dyconstraint, dtype=float)

        # Unknowns: [F_0..F_{N-1}, M_0..M_{N-2}]  (2N-1)
        m = np.zeros((2 * n - 1, 2 * n - 1))
        b = np.zeros(2 * n - 1)

        # Force-sum row (IDL row N-1): sum_k F_k = 0
        m[n - 1, 0:n] = 1.0
        for i in range(n - 1):
            for j in range(i + 1):
                d = xc[i + 1] - xc[j]
                m[i, j] = d ** 3 / 6.0
                m[i, j + n] = d ** 2 / 2.0
                m[i + n, j] = d ** 2 / 2.0
                m[i + n, j + n] = d
            b[i] = yc[i + 1] - yc[0] - dyc[0] * (xc[i + 1] - xc[0])
            b[i + n] = dyc[i + 1] - dyc[0]

        s = np.linalg.lstsq(m, b, rcond=None)[0]

        # Evaluate.  Base line = imposed end slope through first constraint.
        left = xaxis <= xc[n - 1]
        yaxis[left] = dyc[0] * (xaxis[left] - xc[0]) + yc[0]
        for i in range(n - 1):
            seg = (xaxis > xc[i]) & (xaxis <= xc[n - 1])
            dx = xaxis[seg] - xc[i]
            yaxis[seg] += s[i] / 6.0 * dx ** 3 + s[i + n] / 2.0 * dx ** 2
        right = xaxis > xc[n - 1]
        if np.any(right):
            slope = dyc[0]
            for i in range(n - 1):
                d = xc[n - 1] - xc[i]
                slope += s[i] / 2.0 * d ** 2 + s[i + n] * d
            yaxis[right] = yc[n - 1] + slope * (xaxis[right] - xc[n - 1])

        return yaxis

    # Default: unknowns [F_0..F_{N-1}, A]  (N+1)
    m = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    m[0, 0:n] = 1.0                       # sum of forces = 0
    if n > 1:
        m[1, 1:n] = xc[1:n] - xc[0]       # net moment about x0 = 0
    for i in range(n - 1):
        for j in range(i + 1):
            m[i + 2, j] = (xc[i + 1] - xc[j]) ** 3 / 6.0
        m[i + 2, n] = xc[i + 1] - xc[0]
        b[i + 2] = yc[i + 1] - yc[0]

    s = np.linalg.lstsq(m, b, rcond=None)[0]

    left = xaxis <= xc[n - 1]
    yaxis[left] = s[n] * (xaxis[left] - xc[0]) + yc[0]      # A*(x-x0) + Y0
    for i in range(n - 1):
        seg = (xaxis > xc[i]) & (xaxis <= xc[n - 1])
        yaxis[seg] += s[i] / 6.0 * (xaxis[seg] - xc[i]) ** 3
    right = xaxis > xc[n - 1]
    if np.any(right):
        slope = s[n]
        for i in range(n - 1):
            slope += s[i] / 2.0 * (xc[n - 1] - xc[i]) ** 2
        yaxis[right] = yc[n - 1] + slope * (xaxis[right] - xc[n - 1])

    return yaxis
