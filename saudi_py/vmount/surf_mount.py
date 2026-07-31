"""
surf_mount.py — Virtually mount a laser-scanned surface onto graphite spacers.

Ported from IDL surf_mount.pro (Jason Koglin, CAL, July 2002).

Returns the simulated mounted-shell surface: the free-standing laser figure with
the azimuthal deformation required to conform to the spacer constraints removed,
plus (optionally) the spacers' own figure error injected.  The deformation at each
axial position is a beam-bending fit through the spacer azimuths (MountDeflection).

Path used by the virtual-mounting driver (default branch, ``thetastep=2``):
  1. interpolate the (few) azimuthal scans onto a uniform ``thetastep``-degree grid,
  2. low-pass each axial scan and sample its height at the spacer azimuths,
  3. subtract ``MountDeflection(theta, theta_spacer, height_spacer)`` at each z,
  4. add the summed spacer figure ``dr_mountadd`` (from ``sp_dr``).

The ``simple`` and ``fixed`` branches (linear/clamped twist constraint) are ported
for completeness but are not on the driver's path.  The ``bowfit`` keyword is
accepted for call-compatibility but, as in this IDL revision, has no effect.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, July 2002
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_mount.pro; interpol(/quad|/spline) via scipy,
    poly_fit via numpy.polyfit, MountDeflection via mountdeflection.py.
    Returns (dr_mount, theta_abs) so the caller can align grids.
"""
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from .mountdeflection import mountdeflection
from .lowpassfilter import lowpassfilter

_DTOR = np.pi / 180.0


def _interp_quad(x0, y0, x):
    """IDL interpol(y0, x0, x, /quad) — quadratic, extrapolating."""
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if len(x0) < 3:
        f = interp1d(x0, y0, kind='linear', fill_value='extrapolate',
                     bounds_error=False)
    else:
        f = interp1d(x0, y0, kind='quadratic', fill_value='extrapolate',
                     bounds_error=False)
    return f(np.asarray(x, dtype=float))


def _interp_spline(x0, y0, x):
    """IDL interpol(y0, x0, x, /spline) — cubic spline, extrapolating."""
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    order = np.argsort(x0)
    cs = CubicSpline(x0[order], y0[order], extrapolate=True)
    return cs(np.asarray(x, dtype=float))


def surf_mount(theta_laser0, z_laser0, dr_laser0, r0,
               sp_theta=None, sp_z=None, sp_dr=None, nspacers=None,
               hpdavg=None, theta_mnt_eff=None, thetastep=None,
               spacerwidth=None, simple=False, fixed=False,
               bowfit=False, quiet=False, seed=None):
    """Virtually mount a laser surface.  Equivalent to IDL ``surf_mount``.

    Parameters
    ----------
    theta_laser0 : (ntheta,) array-like   azimuthal coordinates [rad]
    z_laser0     : (nz,)     array-like   axial coordinates [mm]
    dr_laser0    : (ntheta, nz) array-like  surface deviations [mm]
    r0           : float   shell radius [mm]
    sp_theta, sp_z, sp_dr : spacer azimuths [rad], axial [mm], figure [mm]
    nspacers     : int     number of spacers (if sp_theta not given)
    theta_mnt_eff: float   angular half-window near a spacer [deg]
    thetastep    : float   re-grid azimuthal step [deg]
    simple, fixed: bool    alternative constraint branches
    seed         : int, optional  RNG seed for the HPDavg branch (unused when
                                  sp_dr is supplied)

    Returns
    -------
    dr_mount : (nscans, nz) ndarray  mounted surface on the (possibly re-gridded) grid
    theta_abs : (nscans,) ndarray    absolute azimuthal coordinates of dr_mount
    """
    theta_laser0 = np.asarray(theta_laser0, dtype=float)
    z_laser0 = np.asarray(z_laser0, dtype=float)
    dr_laser0 = np.asarray(dr_laser0, dtype=float)

    # 1. Optional re-grid to uniform thetastep-degree spacing.
    if thetastep:
        if not quiet:
            print('******* thetastep = ', thetastep)
        nz = len(z_laser0)
        theta_new = (np.arange(int((theta_laser0.max() - theta_laser0.min())
                                   / (thetastep * _DTOR) + 1))
                     * (thetastep * _DTOR) + theta_laser0.min())
        dr_new = np.zeros((len(theta_new), nz))
        for iz in range(nz):
            dr_new[:, iz] = _interp_quad(theta_laser0, dr_laser0[:, iz], theta_new)
        theta_laser0 = theta_new
        dr_laser0 = dr_new

    theta_abs = theta_laser0.copy()

    # 2. Normalise both axes about their means for fitting.
    theta_laser = theta_laser0 - np.mean(theta_laser0)
    z_laser = z_laser0 - np.mean(z_laser0)
    dr_laser = dr_laser0
    nz = len(z_laser)
    nscans = len(theta_laser)

    # 3. Spacer azimuths.
    if sp_theta is None:
        if not nspacers:
            nspacers = 3
        kspacers = (np.arange(nspacers) / (nspacers - 1) * nscans).astype(int)
        theta_spacer = theta_laser[kspacers]
    else:
        sp_theta = np.asarray(sp_theta, dtype=float)
        theta_spacer = np.sort(sp_theta - np.mean(sp_theta))
        nspacers = len(theta_spacer)

    if spacerwidth:
        theta_spacer = np.sort(np.concatenate([
            theta_spacer - spacerwidth / r0 / 2.0,
            theta_spacer + spacerwidth / r0 / 2.0]))

    # 4. Spacer surface (given, or synthetic random slope errors via HPDavg).
    if sp_dr is not None and sp_z is not None:
        sp_dr = np.asarray(sp_dr, dtype=float)
    elif hpdavg:
        rng = np.random.default_rng(seed)
        spacer_slope = hpdavg / 60.0 ** 2 * _DTOR / 2.0 * rng.standard_normal(nspacers)
        sp_dr = np.outer(spacer_slope, z_laser)

    # 5. Inject the spacer figure via summed MountDeflection (dr_mountadd).
    dr_mountadd = None
    if sp_dr is not None and theta_mnt_eff is not None:
        sp_theta_arr = np.asarray(sp_theta, dtype=float)
        acc = np.zeros((nscans, nz))
        for ispacer in range(nspacers):
            kgood = np.where(np.abs(theta_laser - theta_spacer[ispacer])
                             > theta_mnt_eff * _DTOR)[0]
            ngood = len(kgood)
            if ngood > 0:
                dr_add = np.vstack([sp_dr[ispacer, :][np.newaxis, :],
                                    np.zeros((ngood, nz))])
                theta_add = np.concatenate([[sp_theta_arr[ispacer]],
                                            theta_laser[kgood]])
                korder = np.argsort(theta_add)
                theta_add = theta_add[korder]
                dr_add = dr_add[korder, :]
                for iz in range(nz):
                    acc[:, iz] += mountdeflection(theta_laser, theta_add, dr_add[:, iz])
        dr_mountadd = acc

    # 6. Constrain the shell to the spacers.
    if simple or fixed:
        laser_height = np.zeros(nscans)
        laser_slope = np.zeros(nscans)
        for iscan in range(nscans):
            c1, c0 = np.polyfit(z_laser, dr_laser[iscan, :], 1)
            laser_height[iscan] = c0
            laser_slope[iscan] = c1
        height_spacer = _interp_spline(theta_laser, laser_height, theta_spacer)
        slope_spacer = _interp_spline(theta_laser, laser_slope, theta_spacer)
        if fixed:
            dheight = _interp_spline(theta_laser, np.gradient(laser_height, theta_laser),
                                     theta_spacer)
            dslope = _interp_spline(theta_laser, np.gradient(laser_slope, theta_laser),
                                    theta_spacer)
            height_add = mountdeflection(theta_laser, theta_spacer, height_spacer, dheight)
            slope_add = mountdeflection(theta_laser, theta_spacer, slope_spacer, dslope)
        else:
            height_add = mountdeflection(theta_laser, theta_spacer, height_spacer)
            slope_add = mountdeflection(theta_laser, theta_spacer, slope_spacer)
        dr_mount = (dr_laser
                    - np.outer(height_add, np.ones(nz))
                    - np.outer(slope_add, z_laser))
    else:
        # default branch
        dr_mount = np.zeros_like(dr_laser)
        dr_lasersm = np.zeros_like(dr_laser)
        fcut = (z_laser.max() - z_laser.min()) / 25.4 * 4.0
        for itheta in range(nscans):
            dr_lasersm[itheta, :] = lowpassfilter(dr_laser[itheta, :], fcut)
        for iz in range(nz):
            height_spacer = _interp_spline(theta_laser, dr_lasersm[:, iz], theta_spacer)
            dr_mount[:, iz] = (dr_laser[:, iz]
                               - mountdeflection(theta_laser, theta_spacer, height_spacer))

    if dr_mountadd is not None:
        dr_mount = dr_mount + dr_mountadd

    return dr_mount, theta_abs
