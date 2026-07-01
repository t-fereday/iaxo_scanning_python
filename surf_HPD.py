"""
surf_HPD.py — Compute the Half-Power Diameter of a laser-scanned mirror surface.

Ported from IDL surf_HPD.pro.

An on-axis 2-reflection ray-trace is performed for each point on the shell.
The surface residuals ADR are taken as deviations from a perfect conical
surface and are used for the bottom piece; the top piece is taken as a
perfect cone.  (Note: laser-scanner residuals are from a cylinder; the
difference is small for the angles involved.)

HPD values computed [arcsec]:
  HPDtheta  : HALF_WIDTH(2*DRDTHETA/FOCAL, 50)*2  — from azimuthal slopes
  HPDz      : HALF_WIDTH(2*DRDZ, 50)*2            — from axial slopes
  HPDapprox : R0*(MAX(Z)-MIN(Z))/(8*FOCAL^2)      — conic approximation
  HPDtotal  : HALF_WIDTH(SQRT(fx^2+fy^2)/FOCAL, 50)*2 — from focal-plane image
  HPDerror  : scan-to-scan scatter in HPDz

Reference:
  See documentation for HALF_WIDTH.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, May 2002
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_HPD.pro
"""
import numpy as np
from .surf_der import surf_der
from .half_width import half_width


def surf_HPD(vtheta, vz, adr, focal=5200.0, r0=80.0, dsize=None, convol=False):
    """
    Compute the HPD (Half-Power Diameter) of a laser-scanned mirror surface.

    Performs an on-axis 2-reflection ray-trace for each surface point and
    returns the encircled-50 % half-power diameter.

    Equivalent to IDL surf_HPD.

    Parameters
    ----------
    vtheta : (ntheta,) array-like   azimuthal coordinates [rad]
    vz     : (nz,)     array-like   axial coordinates [mm]
    adr    : (ntheta, nz) array-like  surface height residuals [mm]
    focal  : float   telescope focal length [mm]
    r0     : float   shell radius [mm]
    dsize  : float, optional  detector size [mm] (all photons included if None)
    convol : bool    compute W50 via histogram self-convolution

    Returns
    -------
    HPDtotal  : float  total 2-bounce HPD [arcsec]
    HPDtheta  : float  azimuthal HPD [arcsec]
    HPDz      : float  axial HPD [arcsec]
    HPDapprox : float  conic-approximation HPD [arcsec]
    HPDerror  : float  scan-to-scan scatter in axial HPD [arcsec]
    """
    vtheta = np.asarray(vtheta, dtype=float)
    vz     = np.asarray(vz,     dtype=float)
    adr    = np.asarray(adr,    dtype=float)

    imax = len(vtheta)
    jmax = len(vz)

    rad_to_sec = 180.0 * 3600.0 / np.pi
    length     = vz[-1] - vz[0]
    alpha      = r0 / 4.0 / focal

    drdtheta, drdz = surf_der(vtheta, vz, adr)

    # 2-D grids for the ray-trace
    atheta = np.outer(vtheta, np.ones(jmax))
    azaxis = np.outer(np.ones(imax), vz - vz[0] - length / 2.0)

    para = -focal * 2.0 * drdz - azaxis * alpha + adr
    perp = 2.0 * drdtheta

    fx = para * np.cos(atheta) - perp * np.sin(atheta)
    fy = para * np.sin(atheta) + perp * np.cos(atheta)
    fr = np.sqrt(fx ** 2 + fy ** 2)

    if dsize is not None:
        mask = (np.abs(fx) < dsize / 2.0) & (np.abs(fy) < dsize / 2.0)
        drdtheta_use = drdtheta[mask]
        drdz_use     = drdz[mask]
        fr_use       = fr[mask]
    else:
        drdtheta_use = drdtheta
        drdz_use     = drdz
        fr_use       = fr

    count = fr_use.size

    if count > 0:
        if convol:
            from .hist1d import hist1dfit
            _, _, _, W50t, _ = hist1dfit(
                2.0 * drdtheta_use / focal * rad_to_sec,
                noplot=True, convol=True, nbins1=200, nsigma=5.0, zero=True)
            HPDtheta = W50t / np.sqrt(2.0)

            _, _, _, W50z, _ = hist1dfit(
                drdz_use * rad_to_sec,
                noplot=True, convol=True, nbins1=200, nsigma=5.0, zero=True)
            HPDz = W50z / np.sqrt(2.0)
        else:
            HPDtheta = half_width(2.0 * drdtheta_use / focal * rad_to_sec, 50) * 2.0
            HPDz     = half_width(2.0 * drdz_use * rad_to_sec,             50) * 2.0

        HPDtotal = half_width(fr_use / focal * rad_to_sec, 50) * 2.0

        # Scan-to-scan spread in HPDz
        ntheta   = drdz.shape[0]
        HPDerror = 0.0
        for itheta in range(ntheta):
            HPDerror += (HPDz - half_width(2.0 * drdz[itheta, :] * rad_to_sec, 50) * 2.0) ** 2
        HPDerror = np.sqrt(HPDerror) / np.sqrt(ntheta * max(ntheta - 1, 1))
    else:
        print("surf_HPD: no photons in detector")
        HPDtheta = HPDz = HPDtotal = HPDerror = 0.0

    HPDapprox = alpha * length / focal / 2.0 * rad_to_sec

    return HPDtotal, HPDtheta, HPDz, HPDapprox, HPDerror
