"""
surf_create_v7.py — Read raw laser-scanner data and reconstruct a mirror surface.

Ported from IDL surf_create_v7.pro.

Reads all axial scan files for a given shell (named <filename>.txt.1,
.txt.2, …), applies optional calibration and low-pass filtering, reconstructs
the surface height residuals from the slope data, performs alignment and
phase/bow removal, and returns a dict of all key outputs.

Pipeline summary:
  1. Discover and read all .txt.* scan files (read_psd).
  2. Optionally run hardware format conversion (run_format_all — stub).
  3. Low-pass filter each axial scan (lowpass_filter_old).
  4. Sort scans by azimuthal angle and build a uniform grid.
  5. Optionally subtract calibration data (surf_calibrate).
  6. Compute surface slopes from PSD data (Qscale factor).
  7. Reconstruct surface heights from slope data (surf_gen).
  8. Subtract an alignment model (surf_align).
  9. Remove per-scan phase (linear) and bow (quadratic) trends (surf_fit_axial).
 10. Compute HPD metrics (surf_HPD).
 11. Optionally plot 3-D surface maps (surf_plot).
 12. Optionally write output to .npz (surf_write).

Key parameters:
  r0     : shell radius [mm]  (parsed from filename for GSFC glass)
  D0     : PSD distance [mm]  (default 600 mm)
  C0     : cylindrical lens to beamsplitter distance [mm]  (default 155 mm)
  focal  : telescope focal length [mm]  (default 5200 mm for NuSTAR)
  Fcutoff: low-pass cutoff [cycles/inch]  (default 2.54)

Outputs returned as a dict:
  gdr0, gdrdtheta, gdrdz, gsignal, gtheta, gz   — gridded surface arrays
  gdr0_pr, gdr0_br                               — phase-removed, bow-removed
  HPDraw, HPDpr, HPDbr                           — HPD values [arcsec]
  W50, W70                                       — encircled fractions [arcsec]
  HeightRMS, meanbow                             — surface statistics [mm]
  rad_dev, rad_min, rad_max                      — radius statistics [mm]
  skew_param, kurt_param                         — higher moments
  scan_error, filedate, nscans

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
    Sept 2001: added zmin, zmax keywords
    Nov  2001: added filter and thetamin/thetamax keywords
    Dec  2001: changed plotting output
    Sept 2002: GRPRINT keyword replaced by PRINTER
    May  2003: added basedir and outdir keywords
    July 2003: added HPD outputs
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_create_v7.pro; IDL SAVE replaced with
    numpy.savez; IDL SHADE_SURF replaced with matplotlib; goto/data_error
    replaced with DataError exception
"""

import glob
import os
import datetime
import warnings
import numpy as np
import scipy.stats

from .read_psd           import read_psd
from .run_format_all     import run_format_all
from .filters            import lowpass_filter_old
from .calibration_file_match import calibration_file_match
from .surf_calibrate     import surf_calibrate
from .surf_gen           import surf_gen
from .surf_plane         import surf_plane_2d
from .surf_der           import surf_der
from .surf_align         import surf_align
from .surf_center        import surf_center
from .surf_fit_axial     import surf_fit_axial
from .half_width         import half_width
from .surf_HPD           import surf_HPD
from .surf_write         import surf_write
from .surf_plot          import surf_plot
from .hist1d             import hist1dfit


class DataError(Exception):
    """Raised when laser data is insufficient for analysis."""


def surf_create(
    filename,
    fileout=None,
    nscans=None,
    r0=None,
    glass='gsfc',
    newscanner=True,
    basedir=r'C:\NuSTAR\Scan\Laser_Data\GSFC\Scandate_sorted\\',
    outdir=r'C:\NuSTAR\Scan\SXD\\',
    dirsearch=False,
    nocal=False,
    calyear=None,
    caldist=None,
    D0=None,
    C0=None,
    thetastep=None,
    zstep=None,
    thetacut=None,
    zcut=None,
    thetamin=None,
    thetamax=None,
    zmin=None,
    zmax=None,
    zlength=None,
    zcenter=None,
    zfix=False,
    Fcutoff=None,
    nofilter=False,
    noplot=False,
    nowrite=False,
    charsize=2,
    linethick=2.0,
    plotsingle=False,
    nalign=None,
    alignthetacut=None,
    metric=True,
    vscans=None,
    nspacers=None,
    noMount=False,
    WriteMnt=False,
    fitdeg=2,
    theta_mnt_eff=4,
    noalign=False,
    mincounts=None,
    scanrange=None,
    extrascans=False,
    noformat=False,
    VMfile=False,
    quiet=False,
    focal=5200.0,
    convol=False,
):
    """
    Read raw laser-scanner data and reconstruct the mirror surface.

    Parameters mirror IDL surf_create_v7 — see the original .pro header for
    full documentation.  Key outputs are returned as a dict.

    Returns
    -------
    dict with keys:
        gdr0, gdrdtheta, gdrdz, gsignal, gtheta, gz,
        gdr0_pr, gdr0_br,
        HPDraw, HPDpr, HPDbr,
        W50, W70,
        rad_dev, rad_min, rad_max,
        HeightRMS, meanbow,
        skew_param, kurt_param,
        scan_error, filedate,
        nscans (actual number of scans used)
    Returns None if a fatal data error is encountered.
    """
    # ------------------------------------------------------------------
    # Unit labels and conversion factors
    # ------------------------------------------------------------------
    LabHeight = 'microns'
    LabSlope  = 'arcsec'
    LabAz     = 'deg'
    LabAx     = 'cm'
    UnitHeight = 1000.0
    UnitSlope  = 180.0 * 3600.0 / np.pi
    UnitAz     = 180.0 / np.pi
    UnitAx     = 0.1

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    zoffset = 150.0 if newscanner else 0.0

    if glass == 'gsfc':
        if r0 is None:
            try:
                r0 = float(os.path.basename(filename)[1:4]) / 2.0
            except (ValueError, IndexError):
                r0 = 80.0
    else:
        if r0 is None:
            r0 = 80.0

    if fitdeg is None:
        fitdeg = 2
    if theta_mnt_eff is None:
        theta_mnt_eff = 4

    if fileout is None:
        fileout = filename

    # If filename contains a directory component, use it as indir so the
    # caller can pass a full path without also setting basedir.
    _filedir = os.path.dirname(os.path.abspath(filename))
    if os.path.isdir(_filedir):
        indir = _filedir + os.sep
    else:
        indir = basedir
    nofile = False

    if dirsearch:
        pattern = os.path.join(basedir, '**', os.path.basename(filename) + '.mot.10')
        found   = glob.glob(pattern, recursive=True)
        if len(found) >= 1:
            indir  = os.path.dirname(found[0]) + os.sep
            nofile = False
        else:
            nofile = True

    # ------------------------------------------------------------------
    # Count scan files (.txt.*)
    # ------------------------------------------------------------------
    motfiles = sorted(glob.glob(os.path.join(indir, os.path.basename(filename) + '.txt.*')))
    nscans_found = len(motfiles)
    print('motfiles', motfiles)
    nscans_found = int(nscans_found)
    print(f'nscans {nscans_found}')

    # ------------------------------------------------------------------
    # Format raw files (new scanner only, unless noformat)
    # ------------------------------------------------------------------
    # Build vscans_local from scanrange regardless of noformat
    if scanrange is not None:
        vscans_local = list(range(int(scanrange[0]), int(scanrange[1]) + 1))
    else:
        vscans_local = vscans

    if newscanner and not noformat:
        run_format_all(filedir=indir, shellname=os.path.basename(filename),
                       nsnum=nscans_found, mincounts=mincounts,
                       scanrange=scanrange, extrascans=extrascans)

    # ------------------------------------------------------------------
    # File date from .mot.10 file
    # ------------------------------------------------------------------
    motfile_path = os.path.join(indir, os.path.basename(filename) + '.mot.10')
    if os.path.isfile(motfile_path):
        filedate = os.path.getmtime(motfile_path)
    else:
        filedate = 0.0

    upgradedate_dt = datetime.datetime(2009, 3, 11)
    if filedate > 0:
        filedate_dt = datetime.datetime.fromtimestamp(filedate)
    else:
        # No .mot.10 file — mirror IDL behaviour (filedate=0 → epoch 1970 → pre-upgrade → D0=660)
        filedate_dt = datetime.datetime(1970, 1, 1)

    # ------------------------------------------------------------------
    # Instrument distances
    # ------------------------------------------------------------------
    if D0 is None:
        D0 = 660.0 if filedate_dt < upgradedate_dt else 700.0
    if C0 is None:
        C0 = 200.0

    r0 = float(r0)
    D0 = float(D0)
    C0 = float(C0)

    Qscale = 2.0 * D0 * (C0 + r0 / 2.0) / ((D0 - C0 - r0 / 2.0) * r0)

    if not quiet:
        print(f"{filename} found in {indir}")

    # ------------------------------------------------------------------
    # Pre-allocate raw data arrays
    # ------------------------------------------------------------------
    jsize_max = 4092
    jsize     = 0
    scan_error = 0

    raw_n = max(nscans_found, len(vscans_local) if vscans_local else 0, 100)
    aPSDx   = np.zeros((raw_n, jsize_max))
    aPSDy   = np.zeros((raw_n, jsize_max))
    asignal = np.zeros((raw_n, jsize_max))
    atheta  = np.zeros((raw_n, jsize_max))
    azaxis  = np.zeros((raw_n, jsize_max))
    vsignalavg = np.zeros(raw_n)
    nlines     = np.zeros(raw_n, dtype=int)

    # ------------------------------------------------------------------
    # Read scan files
    # ------------------------------------------------------------------
    try:
        if vscans_local is not None:
            _nscans, jsize = _read_vscans(
                vscans_local, indir, filename, jsize_max,
                nofilter, Fcutoff, zoffset,
                aPSDx, aPSDy, asignal, atheta, azaxis, vsignalavg, nlines)
        else:
            _nscans, jsize = _read_sequential(
                nscans_found, indir, filename, jsize_max,
                nofilter, Fcutoff, zoffset,
                aPSDx, aPSDy, asignal, atheta, azaxis, vsignalavg, nlines)
    except DataError as exc:
        print(str(exc))
        return None

    nscans = _nscans

    if not quiet:
        if nofilter:
            print('No filtering')
        else:
            fc = Fcutoff if Fcutoff is not None else 2.54
            print(f'Filtering length [cm] = {2.54/fc:.1f}')

    if nscans < 10:
        print(f'WARNING: less than 10 scans ({nscans} found)')
        _finalise_date(filedate)
        return None

    nlines = nlines[:nscans]
    mean_lines = np.mean(nlines)
    if mean_lines > 0 and ((np.max(nlines) - np.min(nlines)) / mean_lines > 0.4
                            or np.min(nlines) <= 500):
        bad_idx = int(np.argmin(nlines)) + 1
        print(f'Warning: scan data missing? Check number of lines in '
              f'{filename}.txt.{bad_idx} file.')
        print(f'Only {np.min(nlines)} lines found.')
        scan_error += 1

    # ------------------------------------------------------------------
    # Sort by azimuthal angle, trim arrays
    # ------------------------------------------------------------------
    if nscans < 3:
        print(f"File {indir}{filename} not found or too few scans")
        return None

    PSDx   = np.zeros((nscans, jsize))
    PSDy   = np.zeros((nscans, jsize))
    signal = np.zeros((nscans, jsize))
    zaxis  = np.zeros((nscans, jsize))

    raw_theta = atheta[:nscans, 0]
    ktheta    = np.argsort(raw_theta)
    gtheta    = raw_theta[ktheta].astype(float)

    PSDx[:]   = aPSDx[ktheta, :jsize]
    PSDy[:]   = aPSDy[ktheta, :jsize]
    signal[:]  = asignal[ktheta, :jsize]
    zaxis[:]   = azaxis[ktheta, :jsize]
    signalavg  = vsignalavg[ktheta]

    # ------------------------------------------------------------------
    # Azimuthal range
    # ------------------------------------------------------------------
    if thetastep is None:
        thetastep = np.mean(np.diff(gtheta))

    if thetamin is None:
        thetamin = float(np.min(gtheta))
    else:
        thetamin = float(thetamin) * np.pi / 180.0

    if thetamax is None:
        thetamax = float(np.max(gtheta))
    else:
        thetamax = float(thetamax) * np.pi / 180.0

    thetarange = thetamax - thetamin
    kscan = np.where((gtheta >= thetamin) & (gtheta <= thetamax))[0]
    imax  = len(kscan)

    # ------------------------------------------------------------------
    # Axial range from valid data
    # ------------------------------------------------------------------
    if zstep is None:
        zstep = 0.254

    zmin_array = np.zeros(nscans)
    zmax_array = np.zeros(nscans)

    for i in range(nscans):
        good = np.where((zaxis[i, :] > 0) &
                        (zaxis[i, :] < 120.0 + zoffset) &
                        (signal[i, :] > 0.1))[0]
        if len(good) > 0:
            zmin_array[i] = np.min(zaxis[i, good])
            zmax_array[i] = np.max(zaxis[i, good])
        else:
            zmin_array[i] = -1.0e3
            zmax_array[i] =  1.0e3

    zmin_good = np.max(zmin_array[zmin_array > 0.0]) if np.any(zmin_array > 0.0) else 0.0
    zmax_good = np.min(zmax_array[zmax_array > 0.0]) if np.any(zmax_array > 0.0) else 120.0

    if zmin is not None:
        zmin = float(zmin) + zoffset
    if zmax is not None:
        zmax = float(zmax) + zoffset

    if zmin is None:
        zmin = zmin_good
    elif zmin < zmin_good and not zfix:
        zmin = zmin_good
        print('WARNING: zmin too small...modified')

    if zmax is None:
        zmax = zmax_good
    elif zmax > zmax_good and not zfix:
        zmax = zmax_good
        print('WARNING: zmax too large...modified')

    zlength = zmax - zmin

    if not quiet:
        print(f'theta range [deg] = {thetamin/np.pi*180.:7.1f} {thetamax/np.pi*180.:7.1f}')
        print(f'z range [mm] = {zmin-zoffset:7.1f} {zmax-zoffset:7.1f}')

    if zlength < 100.0:
        print('WARNING: zlength is too short. Check laser data files.')
        return None

    # ------------------------------------------------------------------
    # Build uniform axial grid and interpolate
    # ------------------------------------------------------------------
    jmax = int(zlength / zstep + 1)

    gPSDx   = np.zeros((imax, jmax))
    gPSDy   = np.zeros((imax, jmax))
    gsignal = np.zeros((imax, jmax))

    gtheta = gtheta[kscan]
    gz     = np.arange(jmax, dtype=float) * zstep + zmin

    from scipy.interpolate import interp1d

    for i in range(imax):
        iscan_idx = kscan[i]
        good = np.where((zaxis[iscan_idx, :] >= zmin) &
                        (zaxis[iscan_idx, :] <= zmax))[0]
        if len(good) > 1:
            vz_  = zaxis[iscan_idx, good]
            vPx_ = PSDx[iscan_idx, good]
            vPy_ = PSDy[iscan_idx, good]
            vSg_ = signal[iscan_idx, good]
            # Sort by z (may not be monotone after trimming)
            order = np.argsort(vz_, kind='stable')
            vz_ = vz_[order]; vPx_ = vPx_[order]
            vPy_ = vPy_[order]; vSg_ = vSg_[order]

            # Remove duplicate z values (scanner can report same position twice)
            _, unique_idx = np.unique(vz_, return_index=True)
            vz_  = vz_[unique_idx];  vPx_ = vPx_[unique_idx]
            vPy_ = vPy_[unique_idx]; vSg_ = vSg_[unique_idx]

            if len(vz_) < 3:
                continue

            gz_clip = np.clip(gz, vz_.min(), vz_.max())
            gPSDx[i, :]   = interp1d(vz_, vPx_, kind='quadratic')(gz_clip)
            gPSDy[i, :]   = interp1d(vz_, vPy_, kind='quadratic')(gz_clip)
            gsignal[i, :] = interp1d(vz_, vSg_, kind='quadratic')(gz_clip)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    if not nocal:
        if calyear is None:
            if filedate > 0:
                try:
                    calyear = calibration_file_match(filedate)
                except FileNotFoundError:
                    calyear = 'cal_01172024'
            else:
                calyear = 'cal_01172024'

        file_base = calyear

        # Calibration z-range
        if filedate_dt < upgradedate_dt:
            caldist_cal = 100.0
            zmin_cal = -100.0 + zoffset
            zmax_cal =  100.0 + zoffset
        else:
            caldist_cal = 100.0
            zmin_cal = -114.0 + zoffset
            zmax_cal =  120.0 + zoffset

        if caldist is None:
            caldist = caldist_cal

        if not quiet:
            print(f'calibration file = {file_base}')

        PSDx0, PSDy0, signal0, theta0, zaxis0 = surf_calibrate(
            zmin=zmin_cal, zmax=zmax_cal, zstep=zstep,
            file_base=file_base, newscanner=newscanner, noplot=True)

        # Scale calibration to match shell measurement distance
        scale = (D0 + 2.0 * r0) / (D0 + 2.0 * caldist)
        PSDy0 = PSDy0 * scale
        PSDx0 = PSDx0 * scale

        # Interpolate calibration onto gz grid
        gz_clip = np.clip(gz, zaxis0.min(), zaxis0.max())
        PSDx0_gz = interp1d(zaxis0, PSDx0, kind='quadratic',
                             bounds_error=False, fill_value=0.0)(gz_clip)
        PSDy0_gz = interp1d(zaxis0, PSDy0, kind='quadratic',
                             bounds_error=False, fill_value=0.0)(gz_clip)
    else:
        if not quiet:
            print('no calibration')
        PSDx0_gz = np.zeros(jmax)
        PSDy0_gz = np.zeros(jmax)

    # ------------------------------------------------------------------
    # Compute surface slopes from PSD data
    # ------------------------------------------------------------------
    print(f'D0: {D0}')
    nscans = imax

    gdrdtheta = (gPSDx * Qscale - PSDx0_gz[np.newaxis, :]) / (D0 + r0) / 2.0 * r0
    gdrdz     = (gPSDy -          PSDy0_gz[np.newaxis, :]) / (D0 + r0) / 2.0

    gdrdtheta_raw = gdrdtheta.copy()
    gdrdz_raw     = gdrdz.copy()

    # ------------------------------------------------------------------
    # Surface reconstruction
    # ------------------------------------------------------------------
    gdr0, va, vb, vsiga, vsigb, drx, dry, kspacer = surf_gen(
        gtheta, gz, gdrdtheta, gdrdz)

    gdr0_theta = surf_plane_2d(drx, vx=gtheta, vy=gz)
    gdrdtheta_theta, gdrdz_theta = surf_der(gtheta, gz, gdr0_theta)

    gdr0_la = gdr0.copy()

    # Alignment sub-range indices
    ia = 0; ib = nscans - 1
    if alignthetacut is not None:
        while ia < ib and gtheta[ia] < alignthetacut[0] * np.pi / 180.0:
            ia += 1
        while ib > ia and gtheta[ib] > alignthetacut[1] * np.pi / 180.0:
            ib -= 1

    ja = 0; jb = len(gz) - 1

    if not noalign:
        if nalign is None:
            nalign = 1
        nloop = 0
        fit_par = np.zeros(8)
        while True:
            gdr0, fit_par = surf_align(gtheta, gz, gdr0, r0=r0,
                                        ia=ia, ib=ib, ja=ja, jb=jb,
                                        quiet=quiet)
            nloop += 1
            if (abs(fit_par[2]) < 0.00002 and abs(fit_par[3]) < 0.00002) \
                    or nloop >= nalign:
                break
    else:
        adrnew, _, _ = surf_center(gtheta, gz, gdr0, r0=r0)
        gdr0 = surf_plane_2d(adrnew, vx=gtheta, vy=gz)

    gdrdtheta, gdrdz = surf_der(gtheta, gz, gdr0)

    # thetacut / zcut sub-range for output statistics
    ia = 0; ib = nscans - 1
    if thetacut is not None:
        while ia < ib and gtheta[ia] < thetacut[0] * np.pi / 180.0:
            ia += 1
        while ib > ia and gtheta[ib] > thetacut[1] * np.pi / 180.0:
            ib -= 1

    ja = 0; jb = len(gz) - 1
    if zcut is not None:
        while ja < jb and gz[ja] < zcut[0]:
            ja += 1
        while jb > ja and gz[jb] > zcut[1]:
            jb -= 1

    gdr0_z  = surf_plane_2d(dry, vx=gtheta, vy=gz)
    gdrdtheta_z, gdrdz_z = surf_der(gtheta, gz, gdr0_z)

    gdr0_la = surf_plane_2d(gdr0_la, vx=gtheta, vy=gz)
    gdrdtheta_la, gdrdz_la = surf_der(gtheta, gz, gdr0_la)

    gdr0_pr, _          = surf_fit_axial(gdr0,  fitdeg=1)
    gdrdtheta_pr, gdrdz_pr = surf_der(gtheta, gz, gdr0_pr)

    gdr0_br, bow_fitparam = surf_fit_axial(gdr0, fitdeg=2)
    gdrdtheta_br, gdrdz_br = surf_der(gtheta, gz, gdr0_br)

    # ------------------------------------------------------------------
    # Per-scan statistics
    # ------------------------------------------------------------------
    FitHeight  = np.zeros((nscans, 2))
    HPD_arr    = np.zeros(nscans)
    HPDraw_arr = np.zeros(nscans)
    HPDpr_arr  = np.zeros(nscans)
    HPDbr_arr  = np.zeros(nscans)
    RMSheight  = np.zeros(nscans)
    Meanheight = np.zeros(nscans)

    HeightRMS = np.std(gdr0[ia:ib+1, ja:jb+1]) * UnitHeight

    for iscan in range(nscans):
        row  = gz[ja:jb+1]
        data = gdr0[iscan, ja:jb+1]
        coeffs = np.polyfit(row, data, 1)
        FitHeight[iscan, 1] = coeffs[0] * UnitSlope
        FitHeight[iscan, 0] = (np.polyval(coeffs, np.mean(row))) * UnitHeight

        RMSheight[iscan]  = np.std(gdr0[iscan, ja:jb+1]) * UnitHeight
        Meanheight[iscan] = np.mean(gdr0[iscan, ja:jb+1]) * UnitHeight

        HPDraw_arr[iscan] = half_width(gdrdz[iscan, ja:jb+1] * 2.0, 50) * 2.0 * UnitSlope
        HPD_arr[iscan]    = HPDraw_arr[iscan]
        HPDpr_arr[iscan]  = half_width(gdrdz_pr[iscan, ja:jb+1] * 2.0, 50) * 2.0 * UnitSlope
        HPDbr_arr[iscan]  = half_width(gdrdz_br[iscan, ja:jb+1] * 2.0, 50) * 2.0 * UnitSlope

        if abs(HeightRMS) > 0 and abs(RMSheight[iscan] - HeightRMS) / abs(HeightRMS) > 1.0:
            print(f'WARNING: large height deviation at scan #{iscan+1}')
            scan_error += 1

    for iscan in range(nscans):
        low_sig = np.where(gsignal[iscan, ja:jb+1] < 0.1)[0]
        if len(low_sig) > 0.1 * (jb - ja + 1):
            print(f'WARNING: low signal intensity at scan #{iscan+1}')

    # ------------------------------------------------------------------
    # PSF / HPD computation
    # ------------------------------------------------------------------
    if noplot:
        x_hist_raw, y_hist_raw, _, W50r, W70r = hist1dfit(
            gdrdz[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            noplot=True, nbins1=200, nsigma=5.0, zero=True)
        _, _, psf_fit_param, _, _ = hist1dfit(
            gdrdz[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            noplot=True, king=True, nbins1=200, nsigma=5.0, zero=True)
        _, _, _, W50convol_raw, W70convol_raw = hist1dfit(
            gdrdz[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            noplot=True, convol=True, nbins1=200, nsigma=5.0, zero=True)
        _, _, _, W50convol_pr,  W70convol_pr  = hist1dfit(
            gdrdz_pr[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            noplot=True, convol=True, nbins1=200, nsigma=5.0, zero=True)
        _, _, _, W50convol_br,  W70convol_br  = hist1dfit(
            gdrdz_br[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            noplot=True, convol=True, nbins1=200, nsigma=5.0, zero=True)
    else:
        W50convol_raw = W50convol_pr = W50convol_br = 0.0
        W70convol_raw = W70convol_pr = W70convol_br = 0.0
        psf_fit_param = None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if not noplot:
        import matplotlib.pyplot as plt

        # --- 3-D surface maps ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                  subplot_kw={'projection': '3d'})
        fig.suptitle(f'{filename}: 3D surface profile')
        for surface_data, title, ax_3d in zip(
                [gdr0[ia:ib+1, ja:jb+1],
                 gdr0_pr[ia:ib+1, ja:jb+1],
                 gdr0_br[ia:ib+1, ja:jb+1]],
                ['Aligned Raw:', 'Aligned Phase Removed:', 'Aligned Bow Removed:'],
                axes):
            surf_plot(surface_data, gtheta[ia:ib+1], gz[ja:jb+1],
                      title=title, r0=r0, metric=metric,
                      focal=focal, convol=convol,
                      ax=ax_3d, fig=fig)
        plt.tight_layout()
        plt.show()

        # --- 1-D axial profiles ---
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12))
        fig2.suptitle(f'{filename}: 1D surface profile')
        for arr, title_1d, ax_1d in zip(
                [gdr0, gdr0_pr, gdr0_br],
                ['Aligned Raw Profiles:', 'Aligned PR Profiles:', 'Aligned BR Profiles:'],
                axes2):
            for ith in range(ia, ib + 1):
                ax_1d.plot(gz[ja:jb+1], arr[ith, ja:jb+1] * UnitHeight, alpha=0.6)
            ax_1d.set_xlabel('Optic Axis [mm]')
            ax_1d.set_ylabel('Height [µm]')
            ax_1d.set_title(title_1d)
        plt.tight_layout()
        plt.show()

        # --- PSF histograms ---
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
        fig3.suptitle(f'{filename}: Point Spread Function')
        psf_ax = axes3.ravel()

        _, _, psf_fit_param, W50convol_raw, W70convol_raw = hist1dfit(
            gdrdz[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            nbins1=200, nsigma=5.0, zero=True, king=True,
            title='Raw Axial PSF (1B)', xtit=f'Slope [{LabSlope}]', ax=psf_ax[0])
        _, _, _, W50convol_raw, W70convol_raw = hist1dfit(
            gdrdz[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            nbins1=200, nsigma=5.0, zero=True, convol=True,
            title='Raw Axial PSF (2B)', xtit=f'Slope [{LabSlope}]', ax=psf_ax[2])
        _, _, _, W50convol_pr, W70convol_pr = hist1dfit(
            gdrdz_pr[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            nbins1=200, nsigma=5.0, zero=True, king=True,
            title='PR Axial PSF (1B)', xtit=f'Slope [{LabSlope}]', ax=psf_ax[1])
        x_hist_pr_2b, _, _, W50convol_pr, W70convol_pr = hist1dfit(
            gdrdz_pr[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            nbins1=200, nsigma=5.0, zero=True, convol=True,
            title='PR Axial PSF (2B)', xtit=f'Slope [{LabSlope}]', ax=psf_ax[3])
        # B-R reuses P-R's bin boundaries (matches IDL non-noplot path where
        # x_hist_pr_2b is already set when passed to the B-R hist1dfit call).
        _, _, _, W50convol_br, W70convol_br = hist1dfit(
            gdrdz_br[ia:ib+1, ja:jb+1].ravel() * UnitSlope,
            x_hist=x_hist_pr_2b,
            noplot=True, convol=True)
        plt.tight_layout()
        plt.show()

        # --- Individual HPD vs theta ---
        fig4, axes4 = plt.subplots(3, 1, figsize=(8, 10))
        fig4.suptitle(f'{filename}: Individual HPD')
        for arr_hpd, title_hpd, ax_hpd in zip(
                [HPDraw_arr, HPDpr_arr, HPDbr_arr],
                ['Raw HPD', 'PR HPD', 'BR HPD'],
                axes4):
            ax_hpd.plot(gtheta / np.pi * 180.0, arr_hpd * np.sqrt(2.0), 'x')
            ax_hpd.set_xlabel('Theta [deg]')
            ax_hpd.set_ylabel('2B HPD [arcsec]')
            ax_hpd.set_title(title_hpd)
        plt.tight_layout()
        plt.show()

        # --- Fit parameters vs theta ---
        fig5, axes5 = plt.subplots(3, 1, figsize=(8, 10))
        fig5.suptitle(f'{filename}: Individual Fit Parameters')
        axes5[0].plot(gtheta / np.pi * 180.0, RMSheight, 'x')
        axes5[0].set_ylabel(f'Height [{LabHeight}]'); axes5[0].set_title('RMS Height')

        axes5[1].plot(gtheta / np.pi * 180.0,
                      bow_fitparam[:, 1] * zlength * UnitHeight, 'x')
        axes5[1].set_ylabel(f'Height [{LabHeight}]'); axes5[1].set_title('Slope Height')

        axes5[2].plot(gtheta / np.pi * 180.0,
                      bow_fitparam[:, 2] * zlength ** 2 * UnitHeight, 'x')
        axes5[2].set_ylabel(f'Height [{LabHeight}]'); axes5[2].set_title('Bow Height')
        for ax_ in axes5:
            ax_.set_xlabel('Theta [deg]')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    if not nowrite:
        outfile = os.path.join(outdir, os.path.basename(fileout) + '.sxd')
        surf_write(outfile, gdrdtheta, gdrdz, gsignal, gtheta, gz, gdr0, r0)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print(f'Number of axial scans = {nscans:4d}')
    print(f'Number of scan errors = {scan_error:4d}')

    skew_param = float(scipy.stats.skew(gdrdz[ia:ib+1, ja:jb+1].ravel()))
    kurt_param = float(scipy.stats.kurtosis(gdrdz[ia:ib+1, ja:jb+1].ravel()))

    HPDraw = half_width(gdrdz[ia:ib+1, ja:jb+1].ravel() * 2.0, 50) * 2.0 * UnitSlope
    print(f'Raw 2B HPD (convol, gauss) ["] = {W50convol_raw:10.1f} {HPDraw*np.sqrt(2.):10.1f}')

    HPDpr = half_width(gdrdz_pr[ia:ib+1, ja:jb+1].ravel() * 2.0, 50) * 2.0 * UnitSlope
    print(f'P-R 2B HPD (convol, gauss) ["] = {W50convol_pr:10.1f} {HPDpr*np.sqrt(2.):10.1f}')

    HPDbr = half_width(gdrdz_br[ia:ib+1, ja:jb+1].ravel() * 2.0, 50) * 2.0 * UnitSlope
    print(f'B-R 2B HPD (convol, gauss) ["] = {W50convol_br:10.1f} {HPDbr*np.sqrt(2.):10.1f}')

    W50 = [W50convol_raw, W50convol_pr, W50convol_br]
    W70 = [W70convol_raw, W70convol_pr, W70convol_br]

    rad_dev = [np.std(gdr0[ia:ib+1, ja:jb+1]) * UnitHeight,
               np.std(gdr0_pr[ia:ib+1, ja:jb+1]) * UnitHeight,
               np.std(gdr0_br[ia:ib+1, ja:jb+1]) * UnitHeight]
    rad_min = [np.min(gdr0[ia:ib+1, ja:jb+1]) * UnitHeight,
               np.min(gdr0_pr[ia:ib+1, ja:jb+1]) * UnitHeight,
               np.min(gdr0_br[ia:ib+1, ja:jb+1]) * UnitHeight]
    rad_max = [np.max(gdr0[ia:ib+1, ja:jb+1]) * UnitHeight,
               np.max(gdr0_pr[ia:ib+1, ja:jb+1]) * UnitHeight,
               np.max(gdr0_br[ia:ib+1, ja:jb+1]) * UnitHeight]

    print(f'Height RMS [micron] = {rad_dev[0]:6.2f}')

    meanbow = np.mean(bow_fitparam[:, 2]) * zlength ** 2 * UnitHeight
    print(f'Bow height [micron] = {meanbow:6.2f}')
    print(f'Height deviation after BR [micron] = {rad_min[2]:6.2f} ~ {rad_max[2]:6.2f}')
    print('-' * 90)

    # Convert filedate to calendar date string
    if filedate > 0:
        dt = datetime.datetime.fromtimestamp(filedate)
        filedate_str = dt.strftime('%m/%d/%Y')
    else:
        filedate_str = ''

    return dict(
        gdr0=gdr0, gdr0_pr=gdr0_pr, gdr0_br=gdr0_br,
        gdrdtheta=gdrdtheta, gdrdz=gdrdz,
        gsignal=gsignal, gtheta=gtheta, gz=gz,
        HPDraw=HPDraw, HPDpr=HPDpr, HPDbr=HPDbr,
        W50=W50, W70=W70,
        rad_dev=rad_dev, rad_min=rad_min, rad_max=rad_max,
        HeightRMS=HeightRMS, meanbow=meanbow,
        skew_param=skew_param, kurt_param=kurt_param,
        scan_error=scan_error, filedate=filedate_str,
        nscans=nscans,
    )


# ------------------------------------------------------------------
# Helper: read scans by explicit vscans list
# ------------------------------------------------------------------
def _read_vscans(vscans, indir, filename, jsize_max,
                  nofilter, Fcutoff, zoffset,
                  aPSDx, aPSDy, asignal, atheta, azaxis, vsignalavg, nlines):
    jsize = 0
    # Use enumerate so every vscan occupies its fixed slot regardless of read
    # success — mirrors IDL's for-loop where failed reads leave the slot zeroed.
    for iscan, vscan in enumerate(vscans):
        filenumber = str(int(vscan))
        infile = os.path.join(indir, os.path.basename(filename) + f'.txt.{filenumber}')
        vPSDx, vPSDy, vsignal, vtheta, vzaxis, read_error, _ = read_psd(infile, noprint=True)
        if read_error != 0:
            continue   # slot stays zero; nlines[iscan] stays 0
        jlen = len(vzaxis)
        if jlen > jsize_max:
            print(f'read error - too long -- {infile}')
            continue
        nlines[iscan] = jlen
        if jlen > jsize:
            jsize = jlen
        if not nofilter:
            fc = Fcutoff if Fcutoff is not None else 2.54
            span = abs(vzaxis[-1] - vzaxis[0]) / 25.4
            vPSDx = lowpass_filter_old(vPSDx, fc * span)
            vPSDy = lowpass_filter_old(vPSDy, fc * span)
        aPSDx[iscan, :jlen]   = vPSDx
        aPSDy[iscan, :jlen]   = vPSDy
        asignal[iscan, :jlen] = vsignal
        atheta[iscan, :jlen]  = vtheta
        azaxis[iscan, :jlen]  = vzaxis + zoffset
        vsignalavg[iscan]     = np.mean(vsignal)
    return len(vscans), jsize


# ------------------------------------------------------------------
# Helper: read scans sequentially until missing file
# ------------------------------------------------------------------
def _read_sequential(nscans_found, indir, filename, jsize_max,
                      nofilter, Fcutoff, zoffset,
                      aPSDx, aPSDy, asignal, atheta, azaxis, vsignalavg, nlines):
    jsize    = 0
    iscan    = 0
    read_error = 0
    while iscan < nscans_found - 1 and read_error == 0:
        filenumber = str(iscan + 1)
        infile = os.path.join(indir, os.path.basename(filename) + f'.txt.{filenumber}')
        vPSDx, vPSDy, vsignal, vtheta, vzaxis, read_error, _ = read_psd(infile, noprint=True)
        if read_error != 0:
            break
        jlen = len(vzaxis)
        if jlen > jsize_max:
            print(f'read error - too long -- {infile}')
            read_error = 1
            break
        nlines[iscan] = jlen
        if jlen > jsize:
            jsize = jlen
        if not nofilter:
            fc = Fcutoff if Fcutoff is not None else 2.54
            span = abs(vzaxis[-1] - vzaxis[0]) / 25.4
            vPSDx = lowpass_filter_old(vPSDx, fc * span)
            vPSDy = lowpass_filter_old(vPSDy, fc * span)
        aPSDx[iscan, :jlen]   = vPSDx
        aPSDy[iscan, :jlen]   = vPSDy
        asignal[iscan, :jlen] = vsignal
        atheta[iscan, :jlen]  = vtheta
        azaxis[iscan, :jlen]  = vzaxis + zoffset
        vsignalavg[iscan]     = np.mean(vsignal)
        iscan += 1
    return iscan, jsize


def _finalise_date(filedate):
    if filedate > 0:
        dt = datetime.datetime.fromtimestamp(filedate)
        print(dt.strftime('%m/%d/%Y'))
