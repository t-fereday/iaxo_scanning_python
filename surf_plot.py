"""
surf_plot.py — 3-D shaded surface plot with HPD annotation.

Ported from IDL surf_plot.pro.

Plots surface data (deviations from cylindrical) as a 3-D surface map using
matplotlib plot_surface (equivalent to IDL SHADE_SURF).  Image analysis
information (HPD values) is optionally annotated.

Supported unit labels:
  Azimuthal (LabAz)  : 'degrees', 'radians', 'centimeters', 'millimeters',
                        'micrometers', 'inches', 'mils'
  Axial (LabAx)      : 'centimeters', 'millimeters', 'micrometers',
                        'inches', 'mils'
  Height (LabHeight) : 'centimeters', 'millimeters', 'micrometers',
                        'mils', 'inches'
  Slope (LabSlope)   : 'arcminutes', 'arcseconds', 'milliradians'

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Dec 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_plot.pro; IDL SHADE_SURF replaced with
    matplotlib Axes3D.plot_surface
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from .surf_HPD import surf_HPD


_UNIT_AZ = {
    'degrees':     (180.0 / np.pi,  'deg'),
    'radians':     (1.0,            'rad'),
    'centimeters': (None,           'cm'),   # r0-dependent — set at call time
    'millimeters': (None,           'mm'),
    'micrometers': (None,           'um'),
    'inches':      (None,           'in'),
    'mils':        (None,           'mils'),
}
_UNIT_AX = {
    'centimeters': (0.1,    'cm'),
    'millimeters': (1.0,    'mm'),
    'micrometers': (1000.,  'um'),
    'inches':      (1/25.4, 'in'),
    'mils':        (25.4,   'mils'),
}
_UNIT_HEIGHT = {
    'centimeters': (2.54/1000.,  'cm'),
    'millimeters': (25.4/1000.,  'mm'),
    'micrometers': (25.4,        'um'),
    'mils':        (1.0,         'mils'),
    'inches':      (0.001,       'in'),
}
_UNIT_SLOPE = {
    'arcminutes':   (180.*60./np.pi,      "'"),
    'arcseconds':   (180.*3600./np.pi,    '"'),
    'milliradians': (1000.,               'mrad'),
}


def surf_plot(adr, vtheta, vzaxis,
              title='', r0=80.0,
              metric=True,
              LabAz='degrees', LabAx='centimeters', LabHeight='micrometers',
              LabSlope='arcseconds',
              charsize=1, charscale=3,
              convol=False, noText=False, PlotTitle=False,
              focal=5200.0,
              ax=None, fig=None,
              **kwargs):
    """
    Plot a 3-D shaded surface map with optional HPD annotation.

    Equivalent to IDL surf_plot.

    Parameters
    ----------
    adr    : (ntheta, nz) array-like   surface heights [mm]
    vtheta : (ntheta,)   array-like   azimuthal coordinates [rad]
    vzaxis : (nz,)       array-like   axial coordinates [mm]
    title  : str
    r0     : float  shell radius [mm]
    metric : bool   use metric labels if True
    LabAz, LabAx, LabHeight, LabSlope : str  label choices
    convol : bool   use self-convolved HPD
    noText : bool   suppress HPD text annotation
    PlotTitle : bool  plot only title (no HPD stats)
    ax, fig : matplotlib objects  if provided, draw into them

    Returns
    -------
    fig, ax  (matplotlib figure and 3-D axes)
    """
    adr    = np.asarray(adr,    dtype=float)
    vtheta = np.asarray(vtheta, dtype=float)
    vzaxis = np.asarray(vzaxis, dtype=float)

    if metric:
        LabAx     = 'centimeters'
        LabHeight = 'micrometers'

    UnitAz, sLabAz = _UNIT_AZ.get(LabAz, (180./np.pi, 'deg'))
    if UnitAz is None:  # r0-dependent
        scale_map = {'centimeters': 0.1, 'millimeters': 1.0,
                     'micrometers': 1000., 'inches': 1/25.4, 'mils': 1000/25.4}
        UnitAz = scale_map.get(LabAz, 1.0) * r0
        sLabAz = _UNIT_AZ[LabAz][1]

    UnitAx, sLabAx         = _UNIT_AX.get(LabAx,     (0.1,  'cm'))
    UnitHeight, sLabHeight = _UNIT_HEIGHT.get(LabHeight, (25.4, 'um'))

    # IDL's surf_plot multiplies by UnitHeight/25.4*1000 which for "mils" gives ×1
    # We just apply UnitHeight directly since our defaults are metric
    z_plot  = adr    * UnitHeight
    th_plot = vtheta * UnitAz
    ax_plot = vzaxis * UnitAx

    TH, AX = np.meshgrid(th_plot, ax_plot, indexing='ij')

    if fig is None or ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection='3d')

    ax.plot_surface(TH, AX, z_plot, cmap='viridis', alpha=0.85, **kwargs)
    ax.set_xlabel(f"Azimuth [{sLabAz}]")
    ax.set_ylabel(f"Optic Axis [{sLabAx}]")
    ax.set_zlabel(f"Height [{sLabHeight}]")

    if title:
        ax.set_title(title, fontsize=charsize * 10)

    if not noText and not PlotTitle:
        HPDtotal, HPDtheta, HPDz, HPDapprox, HPDerror = surf_HPD(
            vtheta, vzaxis, adr, focal=focal, r0=r0, convol=convol)

        info = (
            f"Shell Radius:    {r0:.0f} mm\n"
            f"HPD azimuth (2B): {HPDtheta*np.sqrt(2):.0f} arcsec\n"
            f"HPD axial (2B):   {HPDz*np.sqrt(2):.0f} arcsec\n"
            f"HPD approx (2B):  {HPDapprox*np.sqrt(2):.0f} arcsec\n"
            f"HPD total (2B):   {HPDtotal*np.sqrt(2):.0f} arcsec\n"
            f"HPD error (2B):   {HPDerror*np.sqrt(2):.0f} arcsec"
        )
        ax.text2D(0.02, 0.98, info, transform=ax.transAxes,
                  fontsize=7, va='top', family='monospace')

    return fig, ax
