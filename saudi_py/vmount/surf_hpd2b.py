"""
surf_hpd2b.py — Analytic 2-bounce Half-Power Diameter of a shell surface.

Ported from IDL surf_HPD2B.pro (Jason Koglin, CAL, Sept 2009).

Performs a single-ray-per-cell on-axis raytrace of a Wolter-I (conical
approximation) optic: the measured surface residual ``adr`` is treated as the
deviation from a perfect cone on the bottom mirror segment, the top segment is
a perfect cone, and the two reflections double the slope (with an extra
``sqrt(2)`` for the statistical double-bounce approximation).  Each cell's
focal-plane displacement is formed and the Half-Power Diameter is the full
width containing 50 % of the rays (``2 * half_width(..., 50)``), in arcsec.

Returns HPDtotal plus the component HPDs (azimuth / axial / conic-approx /
error / spacers), mirroring the IDL keyword outputs.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Sept 2009
    koglin@astro.columbia.edu  (based on surf_HPD, updated for 2B + spacers)
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_HPD2B.pro; IDL DERIV via surf_der (numpy.gradient).
"""
import numpy as np
from ..surf_der import surf_der
from ..half_width import half_width


def surf_hpd2b(vtheta, vz, adr, focal=5600.0, r0=80.0, shell_length=None,
               dsize=None, ksp=None, knotsp=None, reverseconvolve=False):
    """Compute the 2-bounce HPD of a surface.

    Equivalent to IDL ``surf_HPD2B(theta, z, dr, focal=, r0=, ...)``.

    Parameters
    ----------
    vtheta : (ntheta,) array-like   equispaced azimuthal coordinates [rad]
    vz     : (nz,)     array-like   equispaced axial coordinates [mm]
    adr    : (ntheta, nz) array-like  surface deviations from cylindrical [mm]
    focal  : float  telescope focal length [mm] (default 5600)
    r0     : float  mean shell radius [mm] (default 80)
    shell_length : float, optional  shell length for HPDapprox [mm]
    dsize  : float, optional  detector size cut [mm] (default: infinite)
    ksp    : array-like of int, optional  theta indices *on* spacers
    knotsp : array-like of int, optional  theta indices *not* on spacers
    reverseconvolve : bool  use reverse-in-(theta,z) bottom segment instead of
                            the sqrt(2) gaussian approximation

    Returns
    -------
    (HPDtotal, HPDtheta, HPDz, HPDapprox, HPDerror, HPDsp) : floats [arcsec]
    """
    vtheta = np.asarray(vtheta, dtype=float)
    vz = np.asarray(vz, dtype=float)
    adr = np.asarray(adr, dtype=float)
    r0 = float(r0)
    focal = float(focal)

    ntheta = len(vtheta)
    nz = len(vz)

    rad_to_sec = 180.0 * 60.0 * 60.0 / np.pi
    length = vz[nz - 1] - vz[0]
    if shell_length is None:
        shell_length = length
    alpha = r0 / 4.0 / focal

    drdtheta, drdz = surf_der(vtheta, vz, adr)

    atheta = np.tile(vtheta[:, np.newaxis], (1, nz))
    azaxis = np.tile((vz - vz[0] - length / 2.0)[np.newaxis, :], (ntheta, 1))

    if reverseconvolve:
        drdz2b = 2.0 * drdz[::-1, ::-1]
        drdtheta2b = 2.0 * drdtheta[::-1, ::-1]
    else:
        drdz2b = 2.0 * drdz * np.sqrt(2.0)
        drdtheta2b = 2.0 * drdtheta * np.sqrt(2.0)

    para = -focal * drdz2b - azaxis * alpha + adr
    perp = drdtheta2b

    fx = para * np.cos(atheta) - perp * np.sin(atheta)
    fy = para * np.sin(atheta) + perp * np.cos(atheta)
    fr = np.sqrt(fx ** 2 + fy ** 2)

    drdz2b0 = drdz2b.copy()   # full, before spacer masking

    if knotsp is not None:
        knotsp = np.asarray(knotsp, dtype=int)
        drdtheta2b = drdtheta2b[knotsp, :]
        drdz2b = drdz2b[knotsp, :]
        fr = fr[knotsp, :]
        fx = fx[knotsp, :]
        fy = fy[knotsp, :]
    else:
        knotsp = np.arange(ntheta)
    nnotsp = len(knotsp)

    if dsize is not None:
        gmask = (np.abs(fx) < dsize / 2.0) & (np.abs(fy) < dsize / 2.0)
        count = int(np.sum(gmask))
        if count > 0:
            drdtheta2b = drdtheta2b[gmask]
            drdz2b = drdz2b[gmask]
            fr = fr[gmask]
    else:
        count = drdtheta2b.size

    if count > 0:
        hpdtheta = half_width(drdtheta2b / focal * rad_to_sec, 50) * 2.0
        hpdz = half_width(drdz2b * rad_to_sec, 50) * 2.0
        hpdtotal = half_width(fr / focal * rad_to_sec, 50) * 2.0
        hpderror = 0.0
        for itheta in range(nnotsp):
            row = drdz2b0[knotsp[itheta], :]
            hpderror += (hpdz - half_width(row * rad_to_sec, 50) * 2.0) ** 2
        denom = nnotsp * (nnotsp - 1)
        hpderror = np.sqrt(hpderror) / np.sqrt(denom) if denom > 0 else 0.0
        hpdsp = 0.0
        if ksp is not None:
            ksp = np.asarray(ksp, dtype=int)
            hpdsp = half_width(drdz2b0[ksp, :] * rad_to_sec, 50) * 2.0
    else:
        print('surf_hpd2b: no photons in detector')
        hpdtheta = hpdz = hpdtotal = hpderror = hpdsp = 0.0

    hpdapprox = alpha * shell_length / focal / 2.0 * rad_to_sec

    return hpdtotal, hpdtheta, hpdz, hpdapprox, hpderror, hpdsp
