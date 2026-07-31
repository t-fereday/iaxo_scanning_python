"""
load_optics_geometry.py — Per-layer radius / cone-angle lookup.

Ports the ``readcol, 'BabyIAXO_geometry.txt', ...`` table read and the
``Radius_center(103-layer)`` / ``Theta_deg(103-layer)`` indexing from
LoadLaserSampleFiles.pro.

IMPORTANT — two distinct radii (do not conflate):
  * ``r0`` (drives the HPD via alpha = r0/4/focal, and the spacer spacing) comes
    from the *filename diameter* (SampleDiameter/2), NOT from this table.
  * ``layer_r`` here is the layer's *design* radius.  In the current IDL it feeds
    only ``dr_conic``/``adr_conic``, which are computed but not used downstream
    (drmnt = drraw - drerrhp - drerrlp), so this table does not affect today's
    _Mount output.  The loader is provided for completeness / future use and
    degrades gracefully when ``BabyIAXO_geometry.txt`` is absent (it is not
    shipped in the IDL tree — it lives in the lab's scan/SXD/).

Modification History:
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port of the BabyIAXO_geometry.txt read; IDL readcol -> numpy.loadtxt.
"""
import os
import numpy as np


def load_geometry(path):
    """Read BabyIAXO_geometry.txt.

    Columns (SKIPLINE=1, COMMENT='#'):
        Theta_deg  Radius_front  Radius_center  mlength  pore_h

    Returns
    -------
    dict of column arrays, or None if the file does not exist.
    """
    if not path or not os.path.exists(path):
        return None
    cols = np.loadtxt(path, comments='#', skiprows=1, unpack=True)
    theta_deg, radius_front, radius_center, mlength, pore_h = cols[:5]
    return dict(theta_deg=theta_deg, radius_front=radius_front,
                radius_center=radius_center, mlength=mlength, pore_h=pore_h)


def layer_geometry(layer, geom):
    """Return (layer_r_mm, layer_theta_rad) for a layer, or (None, None).

    Mirrors IDL: layer_r = Radius_center(103-layer)*1000 [mm];
                 layer_Theta = Theta_deg(103-layer)*!pi/180.
    """
    if geom is None:
        return None, None
    idx = 103 - int(layer)
    if idx < 0 or idx >= len(geom['radius_center']):
        return None, None
    layer_r = float(geom['radius_center'][idx]) * 1000.0
    layer_theta = float(geom['theta_deg'][idx]) * np.pi / 180.0
    return layer_r, layer_theta
