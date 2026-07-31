"""
virtual_mount.py — Virtual-mounting simulation orchestrator.

Ports the ``MakePlots`` block of LoadLaserSampleFiles.pro (lines 253-412) plus
Plot_LaserMountSim.pro: read one laser-scanned shell (.sxd / .sxd.npz), split the
figure into axial low- and high-pass bands, build the 5-spacer constraint with a
synthetic random spacer figure error, virtually mount each band (surf_mount ->
MountDeflection), and produce the six-panel Mount figure with 2-bounce HPDs.

Panels (matching the IDL 2x3 layout):
    Raw Low Pass  | Raw High Pass
    Mount Low Pass| Mount High Pass
    Raw           | Mounted

Randomness (unseeded by default, matching IDL ``randomu(seed,...)``):
  the synthetic spacer figure error ``sp_drerr`` — see the marked block below.
  It makes the Mount-High-Pass and Mounted panels fluctuate run-to-run; the
  Raw / Raw-LP / Raw-HP / Mount-LP panels are deterministic.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory (LoadLaserSampleFiles /
    Plot_LaserMountSim)  koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port of the MakePlots mount path; IDL randomu -> numpy Generator
    (unseeded by default), rebin/interpol via numpy/scipy, plots via matplotlib
    with the HPD annotation placed clear of the contour grid.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from ..surf_der import surf_der
from ..surf_write import surf_write
from .sxd_io import surf_read
from .lowpassfilter import lowpassfilter
from .surf_mount import surf_mount
from .surf_plot2b import surf_plot2b
from .load_optics_geometry import load_geometry, layer_geometry

_DTOR = np.pi / 180.0

# ---------------------------------------------------------------------------
# Default output directory for the Python mount figures (pdf/png), anchored to
# the repo root (parent of saudi_py/) — the Python counterpart of idl_out/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_OUT_DIR = os.path.join(_REPO_ROOT, 'py_out')

# Panel rendering styles (see surf_plot2b).  'all' renders one figure of each,
# from a single physics run, so they show the same realization.
_PANEL_STYLES = ('idl', 'heatmap', 'labeled')
# ---------------------------------------------------------------------------


def parse_sample_name(sxd_path):
    """Parse the shell metadata from the .sxd filename (IDL strmid convention)."""
    base = os.path.basename(sxd_path)
    for ext in ('.sxd.npz', '.npz', '.sxd'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    diameter = None
    layer = None
    try:
        diameter = int(base[1:4])
    except (ValueError, IndexError):
        pass
    try:
        layer = int(base[9:12])
    except (ValueError, IndexError):
        pass
    return base, diameter, layer


def _rebin_1d(vec, n):
    """IDL rebin(vec, n) — expand (linear interp) or shrink (block average)."""
    vec = np.asarray(vec, dtype=float)
    m = len(vec)
    if n == m:
        return vec.copy()
    if n > m:                       # expand by linear interpolation
        return np.interp(np.linspace(0, m - 1, n), np.arange(m), vec)
    factor = m // n                 # shrink by block averaging
    return vec[:n * factor].reshape(n, factor).mean(axis=1)


def _align_to_theta(dr_src, theta_src, theta_dst):
    """Interpolate dr_src (theta_src, nz) onto theta_dst per z-column.

    Identity when the grids coincide (surf_mount's thetastep=2 re-grid matches a
    2-degree-sampled input); interpolation only kicks in if they differ.
    """
    theta_src = np.asarray(theta_src, dtype=float)
    theta_dst = np.asarray(theta_dst, dtype=float)
    if len(theta_src) == len(theta_dst) and np.allclose(theta_src, theta_dst, atol=1e-6):
        return dr_src
    nz = dr_src.shape[1]
    out = np.zeros((len(theta_dst), nz))
    kind = 'quadratic' if len(theta_src) >= 3 else 'linear'
    for iz in range(nz):
        f = interp1d(theta_src, dr_src[:, iz], kind=kind,
                     fill_value='extrapolate', bounds_error=False)
        out[:, iz] = f(theta_dst)
    return out


def virtual_mount(sxd_path, layer=None, r0=None, focal=5600.0, out=None,
                  seed=None, quiet=False, geometry_file=None,
                  hpd_mount=30.0, length_scale_min0=5.0,
                  length_scale_range0=25.0, dust_index=1.0,
                  make_plot=True, sp_drerr=None, show=True, style='all'):
    """Run the virtual-mounting simulation on one shell.

    Parameters
    ----------
    sxd_path : str    path to the shell surface (.sxd or .sxd.npz)
    layer    : int, optional  layer number (default: parsed from filename)
    r0       : float, optional  shell radius [mm] (default: filename diameter/2)
    focal    : float  focal length [mm] (default 5600, BabyIAXO)
    out      : str, optional  output figure path (default: <name>_Mount.pdf beside input)
    seed     : int, optional  RNG seed for the spacer error (default: unseeded, per IDL)
    geometry_file : str, optional  path to BabyIAXO_geometry.txt (for dr_conic; unused)
    make_plot : bool  render and save the 6-panel figure
    sp_drerr : (nspacers, nz) array, optional
        Inject a spacer figure-error realization instead of drawing one from
        the RNG (used to reproduce an IDL run exactly — see
        idl_out/compare_py_idl.py). Default None: draw randomly as usual.
    show     : bool  pop up the interactive matplotlib window after saving
        (like surf_create_v7's plt.show()); False = headless save-only (Agg).
    style    : str or sequence  panel rendering: 'idl' (plain IDL-style
        contours), 'heatmap' (filled diverging map + colorbar), 'labeled'
        (contours with values on the lines), or 'all' (default: one figure of
        each).  The simulation runs once, so every figure shows the same
        realization; only the drawing differs.

    Returns
    -------
    dict  per-panel HPDs (total/axial/spacers) and metadata.  'figures' maps
    each style to its (pdf, png) paths; 'figure'/'figure_png' are the first.
    """
    name, diameter, file_layer = parse_sample_name(sxd_path)
    if layer is None:
        layer = file_layer if file_layer is not None else 1

    surf = surf_read(sxd_path)
    theta = surf.theta
    z = surf.z
    dr = surf.dr
    signal = surf.signal
    if r0 is None:
        r0 = (diameter / 2.0) if diameter is not None else surf.r0

    ntheta = len(theta)
    nz = len(z)
    zspan = z.max() - z.min()

    # --- constants (LoadLaserSampleFiles MakePlots) ---
    nspacers = 5
    length_scale_min = length_scale_min0            # 5 mm
    length_scale_range = length_scale_range0 / 2.0  # 12.5 mm
    shell_length0 = 225.0
    nvals = int(shell_length0 / (length_scale_range / 2.0 + length_scale_min) * 1.5)

    # --- split into axial low-/high-pass bands ---
    drraw = dr
    dr00 = dr
    drlp = np.zeros_like(dr)
    fcut_in = 2.54 / 2.0 * zspan / 25.4
    fcut_out = 0.254 * zspan / 25.4
    for itheta in range(ntheta):
        inner = lowpassfilter(dr[itheta, :], fcut_in, weedsigma=5, weedsmoothscale=5)
        drlp[itheta, :] = lowpassfilter(inner, fcut_out)
    drhp = dr - drlp

    # --- spacer azimuths (layer >= 66 uses the narrow +-15 set) ---
    if layer >= 66:
        sp_theta = np.array([-15.0 + 2.4 / r0 / _DTOR, -7.5, 0.0, 7.5,
                             15.0 - 2.4 / r0 / _DTOR]) * _DTOR
    else:
        sp_theta = np.array([-30.0 + 2.4 / r0 / _DTOR, -15.0, 0.0, 15.0,
                             30.0 - 2.4 / r0 / _DTOR]) * _DTOR

    # optional geometry table (feeds only the currently-unused dr_conic)
    geom = load_geometry(geometry_file)
    layer_r, layer_theta = layer_geometry(layer, geom)  # noqa: F841 (dr_conic unused)

    # --- build per-spacer profiles + synthetic random figure error ---
    rng = np.random.default_rng(seed)   # unseeded when seed is None (matches IDL)
    sp_drerr_in = sp_drerr              # injected realization (or None)
    if sp_drerr_in is not None:
        sp_drerr_in = np.asarray(sp_drerr_in, dtype=float)
        if sp_drerr_in.shape != (nspacers, nz):
            raise ValueError(f'sp_drerr shape {sp_drerr_in.shape} != '
                             f'({nspacers}, {nz})')
    sp_drerr = np.zeros((nspacers, nz))
    sp_drhp = np.zeros((nspacers, nz))
    sp_drlp = np.zeros((nspacers, nz))
    vksp = np.zeros(ntheta, dtype=int)
    delta_height_stdev = (hpd_mount / 60.0 * (length_scale_range / 2.0 + length_scale_min)
                          / 10.0 / 2.0 / 1000.0)   # mm
    for ispacer in range(nspacers):
        spwidth = 2.2 / r0 / _DTOR
        ksp_idx = np.where(np.abs(theta - np.mean(theta) - sp_theta[ispacer])
                           < spwidth * _DTOR)[0]
        while len(ksp_idx) == 0:
            spwidth += 0.3
            ksp_idx = np.where(np.abs(theta - np.mean(theta) - sp_theta[ispacer])
                               < spwidth * _DTOR)[0]
        vksp[ksp_idx] = 1

        if sp_drerr_in is not None:
            # injected realization (e.g. from an IDL run) — bypass the RNG
            sperr = sp_drerr_in[ispacer, :]
        else:
            # >>> RANDOMNESS (unseeded like IDL) — synthetic spacer figure error <<<
            steps = rng.random(nvals - 1) * length_scale_range + length_scale_min
            zvals = np.concatenate([[0.0], np.cumsum(steps)]) + z.min()
            drvals = np.abs(rng.standard_normal(nvals) * delta_height_stdev) ** dust_index
            # spline drvals(zvals) onto a 10x-oversampled z grid, then block-average back
            z_fine = _rebin_1d(z, nz * 10)
            order = np.argsort(zvals)
            sperr_fine = CubicSpline(zvals[order], drvals[order], extrapolate=True)(z_fine)
            sperr = _rebin_1d(sperr_fine, nz)
            # NOTE: IDL leaves the HPD rescale of sperr commented out — we do too.
            # <<< end randomness >>>

        count = len(ksp_idx)
        sphp = lowpassfilter(np.sum(drhp[ksp_idx, :], axis=0) / count, zspan / 25.4 * 4.0)
        sp_drerr[ispacer, :] = sperr
        sp_drhp[ispacer, :] = sphp
        sp_drlp[ispacer, :] = np.sum(drlp[ksp_idx, :], axis=0) / count

    ksp = np.where(vksp == 1)[0]
    knotsp = np.where(vksp == 0)[0]

    # --- virtually mount each band ---
    # bowfit=True matches the IDL calls (/bowfit in LoadLaserSampleFiles 354/363)
    drerrhp_g, th_hp = surf_mount(theta, z, drhp, r0, nspacers=nspacers,
                                  theta_mnt_eff=10.0 / r0 / _DTOR,
                                  sp_theta=sp_theta, sp_z=z, sp_dr=sp_drerr,
                                  thetastep=2, bowfit=True, quiet=quiet)
    drerrlp_g, th_lp = surf_mount(theta, z, drlp, r0, nspacers=nspacers,
                                  theta_mnt_eff=7.5 / r0 / _DTOR,
                                  sp_theta=sp_theta, sp_z=z, sp_dr=sp_drlp * 0.0,
                                  thetastep=2, bowfit=True, quiet=quiet)
    drerrhp = _align_to_theta(drerrhp_g, th_hp, theta)
    drerrlp = _align_to_theta(drerrlp_g, th_lp, theta)
    drmnt = drraw - drerrhp - drerrlp

    # mounted-simulation surface (as IDL writes <name>_mntsim.sxd)
    drdtheta_m, drdz_m = surf_der(theta, z, drmnt)

    # --- six panels ---
    panels = [
        ('Raw Low Pass', drlp),
        ('Raw High Pass', drhp),
        ('Mount Low Pass', drerrlp),
        ('Mount High Pass', drerrhp),
        ('Raw', dr00),
        ('Mounted', drmnt),
    ]

    results = {'name': name, 'layer': layer, 'r0': float(r0), 'focal': float(focal),
               'panels': {},
               # stage arrays for validation tooling (idl_out/compare_py_idl.py)
               'arrays': {'theta': theta, 'z': z, 'dr': dr,
                          'drlp': drlp, 'drhp': drhp,
                          'sp_drerr': sp_drerr, 'sp_drhp': sp_drhp,
                          'sp_drlp': sp_drlp, 'ksp': ksp, 'knotsp': knotsp,
                          'drerrhp': drerrhp, 'drerrlp': drerrlp,
                          'drmnt': drmnt}}

    # IDL Plot_LaserMountSim passes theta-mean(theta), z-mean(z) to surf_plot2b so
    # the axes centre on zero and the (centred) spacer lines line up.  The HPDs are
    # invariant to this constant shift (fr = sqrt(fx^2+fy^2) is rotation-invariant).
    theta_c = theta - np.mean(theta)
    z_c = z - np.mean(z)

    # Requested panel style(s).  The physics above ran ONCE, so every style shows
    # the same realization (important: the spacer error is random).
    if isinstance(style, str):
        styles = list(_PANEL_STYLES) if style == 'all' else [style]
    else:
        styles = list(style)
    for sty in styles:
        if sty not in _PANEL_STYLES:
            raise ValueError(f'style {sty!r} not one of '
                             f'{_PANEL_STYLES + ("all",)}')

    def _draw_panels(sty='labeled', axes_map=None):
        """Draw the six panels (if axes given) and return their HPDs."""
        hpds = {}
        for title, surf_arr in panels:
            ax_plot, ax_text = (axes_map or {}).get(title, (None, None))
            hp = surf_plot2b(surf_arr, theta_c, z_c, ax=ax_plot, ax_text=ax_text,
                             title=title, r0=r0, focal=focal, sp_theta=sp_theta,
                             ksp=ksp, knotsp=knotsp, metric=True, style=sty)
            hpds[title] = {
                'HPDtotal': hp[0], 'HPDtheta': hp[1], 'HPDz': hp[2],
                'HPDapprox': hp[3], 'HPDerror': hp[4], 'HPDsp': hp[5],
            }
        return hpds

    figures = {}
    if make_plot:
        # Interactive backend (like surf_create_v7): the figures pop up after
        # the run via plt.show() unless show=False. Agg is only forced when
        # explicitly requested (headless/batch use).
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        # panel (row, col) -> (plot gridspec, text gridspec)
        order = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        for sty in styles:
            fig = plt.figure(figsize=(11, 12))
            gs = GridSpec(3, 4, width_ratios=[3, 2, 3, 2], figure=fig,
                          hspace=0.35, wspace=0.25)
            axes_map = {}
            for (title, _), (r, c) in zip(panels, order):
                axes_map[title] = (fig.add_subplot(gs[r, 2 * c]),
                                   fig.add_subplot(gs[r, 2 * c + 1]))
            fig.suptitle(f'{name}   [{sty}]', fontsize=12, y=0.98)
            hpds = _draw_panels(sty, axes_map)
            if not results['panels']:      # identical for every style
                results['panels'] = hpds

            if out is not None and len(styles) == 1:
                path = out
            else:
                os.makedirs(PY_OUT_DIR, exist_ok=True)
                path = os.path.join(PY_OUT_DIR, f'{name}_Mount_{sty}.pdf')
            fig.savefig(path, dpi=150)
            png = os.path.splitext(path)[0] + '.png'
            fig.savefig(png, dpi=150)
            figures[sty] = (path, png)
            if not quiet:
                print(f'mount figure [{sty:<7}] saved to: {path}')
        results['figures'] = figures
        results['figure'], results['figure_png'] = figures[styles[0]]
    else:
        results['panels'] = _draw_panels()

    if not quiet:
        print(f'\n{name}  (layer {layer}, r0 {r0:.1f} mm, focal {focal:.0f} mm)')
        print(f'{"panel":<16}{"total":>8}{"axial":>8}{"spacers":>9}')
        for title, _ in panels:
            p = results['panels'][title]
            print(f'{title:<16}{p["HPDtotal"]:>8.0f}{p["HPDz"]:>8.0f}{p["HPDsp"]:>9.0f}')

    if make_plot:
        import matplotlib.pyplot as plt
        if show:
            plt.show()   # shows every style's window; blocks until all are closed
        plt.close('all')

    return results
