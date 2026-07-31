"""
surf_plot2b.py — One contour panel of a mounted/raw surface, with HPD annotation.

Ported from IDL surf_plot2B.pro (Jason Koglin, CAL, Sept 2009), the ``/cont``
mode as used by Plot_LaserMountSim.pro.  Axes are Azimuth [deg] × Optic Axis
[cm]; vertical spacer lines are overlaid and the 2-bounce HPDs come from
surf_hpd2b.  Three panel styles are available via ``style=``:

  'idl'     — as the original IDL: nlev plain black contour lines, no labels.
  'heatmap' — filled diverging map (RdBu_r) centred on zero + colorbar in µm.
  'labeled' — topo contours with the height values written on the lines (default).

The height data and every HPD number are identical across the three; only the
rendering differs.

Deviation from the IDL layout (intentional): IDL overprints the red HPD text on
top of the contour grid; here the annotation is drawn in a reserved text axes
(``ax_text``) beside the plot so it never covers the data — the same fix applied
to surf_plot.py in the surf_create port.  All fields/values match the IDL text.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Sept 2009
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_plot2B.pro (contour mode); IDL CONTOUR ->
    matplotlib contour; annotation moved to a reserved axes.
"""
import numpy as np
from .surf_hpd2b import surf_hpd2b


def surf_plot2b(adr, vtheta, vzaxis, ax=None, ax_text=None,
                title='', r0=160.0, focal=5600.0, shell_length=None,
                sp_theta=None, ksp=None, knotsp=None, metric=True,
                nlev=18, tcolor='#cc0000', style='labeled'):
    """Draw one contour panel and return its 2-bounce HPDs.

    Equivalent to IDL ``surf_plot2B, adr, vtheta, vzaxis, /cont, /metric``.

    Parameters
    ----------
    adr    : (ntheta, nz) array-like  surface deviations [mm]
    vtheta : (ntheta,) array-like     azimuthal coordinates [rad]
    vzaxis : (nz,)     array-like     axial coordinates [mm]
    ax      : matplotlib Axes for the contour (created if None)
    ax_text : matplotlib Axes for the annotation (optional)
    title  : str
    r0     : float  shell radius [mm]
    focal  : float  focal length [mm]
    sp_theta : array-like  spacer azimuths [rad] (drawn as vertical lines)
    ksp, knotsp : spacer / non-spacer theta index masks (passed to surf_hpd2b)
    metric : bool  metric axis labels (cm / um)
    nlev   : int   number of contour levels

    Returns
    -------
    (HPDtotal, HPDtheta, HPDz, HPDapprox, HPDerror, HPDsp) : floats [arcsec]
    """
    adr = np.asarray(adr, dtype=float)
    vtheta = np.asarray(vtheta, dtype=float)
    vzaxis = np.asarray(vzaxis, dtype=float)

    # Unit conversions matching IDL /metric contour mode:
    #   height mm -> um (adr*UnitHeight/25.4*1000 with UnitHeight=25.4 -> adr*1000)
    #   azimuth rad -> deg (UnitAz = 180/pi)
    #   optic axis mm -> cm (vzaxis*UnitAx/25.4 with UnitAx=2.54 -> vzaxis*0.1)
    unit_az = 180.0 / np.pi
    slab_az = 'degree'
    unit_ax_over_254 = 2.54 / 25.4          # -> cm
    slab_ax = 'cm'

    th_deg = vtheta * unit_az
    z_cm = vzaxis * unit_ax_over_254
    height_um = adr * 1000.0

    if ax is not None:
        Xg, Yg = np.meshgrid(th_deg, z_cm)      # shape (nz, ntheta)

        if style == 'idl':
            # Faithful to IDL surf_plot2B /cont: nlev plain black contour lines,
            # no labels, no colour (negatives come out dashed, as in IDL).
            ax.contour(Xg, Yg, height_um.T, levels=nlev,
                       colors='k', linewidths=0.4)
        else:
            # Robust symmetric range (98th pct) so a few extreme pixels don't
            # swallow the scale / all the levels.
            vmax = float(np.percentile(np.abs(height_um), 98.0))
            if vmax <= 0.0:
                vmax = float(np.abs(height_um).max()) or 1e-12
            from matplotlib.ticker import MaxNLocator

            if style == 'heatmap':
                # Filled diverging map centred on zero + colorbar in µm.
                levels = np.linspace(-vmax, vmax, 41)
                im = ax.contourf(Xg, Yg, height_um.T, levels=levels,
                                 cmap='RdBu_r', extend='both')
                cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
                cbar.set_ticks([t for t in
                                MaxNLocator(nbins=5, symmetric=True)
                                .tick_values(-vmax, vmax) if -vmax <= t <= vmax])
                cbar.ax.set_title('µm', fontsize=6.5)
                cbar.ax.tick_params(labelsize=6)
            else:
                # 'labeled': topo contours with the values written on the lines.
                line_levels = [t for t in
                               MaxNLocator(nbins=10, symmetric=True)
                               .tick_values(-vmax, vmax) if -vmax <= t <= vmax]
                cs = ax.contour(Xg, Yg, height_um.T, levels=line_levels,
                                colors='k', linewidths=0.5)
                ax.clabel(cs, fontsize=6, fmt='%g', inline=True, inline_spacing=3)

        ax.set_xlabel(f'Azimuth [{slab_az}]', fontsize=7)
        ax.set_ylabel(f'Optic Axis [{slab_ax}]', fontsize=7)
        ax.tick_params(labelsize=6)
        if sp_theta is not None:
            for spt in np.asarray(sp_theta, dtype=float):
                ax.plot([spt * unit_az] * 2, [z_cm.min(), z_cm.max()],
                        'k-', lw=1.0)
        ax.set_xlim(th_deg.min(), th_deg.max())
        ax.set_ylim(z_cm.min(), z_cm.max())

    # HPDs from the analytic 2-bounce estimator.
    hpdtotal, hpdtheta, hpdz, hpdapprox, hpderror, hpdsp = surf_hpd2b(
        vtheta, vzaxis, adr, focal=focal, r0=r0, shell_length=shell_length,
        ksp=ksp, knotsp=knotsp)

    shell_len = (vzaxis.max() - vzaxis.min()) * unit_ax_over_254
    shell_arc = (vtheta.max() - vtheta.min()) * unit_az

    lines = [title]
    if style == 'labeled':
        lines.append("Contour values:   µm")
    lines += [
        f"Shell Radius:     {round(r0):d} mm",
        f"Plot Elements:    {len(vtheta)}x{len(vzaxis)}",
        f"Shell Length:     {shell_len:.1f} {slab_ax}",
        f"Shell Arc:        {shell_arc:.1f} {slab_az}",
        f"HPD azimuth (2B): {round(hpdtheta):d} arcsec",
        f"HPD axial (2B):   {round(hpdz):d} arcsec",
        f"HPD approx (2B):  {round(hpdapprox):d} arcsec",
        f"HPD total (2B):   {round(hpdtotal):d} arcsec",
        f"HPD error (2B):   {round(hpderror):d} arcsec",
    ]
    if ksp is not None:
        lines.append(f"HPD spacers (2B): {round(hpdsp):d} arcsec")

    if ax_text is not None:
        ax_text.axis('off')
        ax_text.text(0.0, 1.0, '\n'.join(lines), transform=ax_text.transAxes,
                     va='top', ha='left', fontsize=6.5, family='monospace',
                     color=tcolor)
    elif ax is not None:
        # No reserved axes given — fall back to a boxed inset in the corner.
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=6, family='monospace', color=tcolor,
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec=tcolor, lw=0.5,
                          alpha=0.85))

    return hpdtotal, hpdtheta, hpdz, hpdapprox, hpderror, hpdsp
