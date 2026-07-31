"""
half_width.py — Binary-search encircled-fraction half-width calculator.

Ported from IDL half_width.pro.

Calculates the half-width of the range centred about zero in which
EncircledFraction percent of the (optionally weighted) values of Event
are contained.  The search is performed via a binary search over the
absolute values of Event.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
    May 2001: added WEIGHT keyword
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL half_width.pro
"""
import numpy as np


def half_width(event0, encircled_fraction, weight=None,
               goal_precision=0.01, n_loop_max=500):
    """
    Return the half-width of the range centred about zero that contains
    *encircled_fraction* percent of the (optionally weighted) data.

    Faithful port of IDL half_width.pro using a binary search.

    Parameters
    ----------
    event0 : array-like
        1-D data array (absolute value is used).
    encircled_fraction : float
        Target encircled fraction in percent (e.g. 50 for HPD).
    weight : array-like, optional
        Per-element weights (default: uniform unity weights).
    goal_precision : float
        Convergence criterion on |EncircledEvents - GoalEvents| / GoalEvents.
    n_loop_max : int
        Maximum binary-search iterations.

    Returns
    -------
    float
        Half-width value.  Returns 999.0 if the search fails.
    """
    event0 = np.asarray(event0, dtype=float)
    event  = np.abs(event0.ravel())

    if weight is None:
        weight = np.ones(len(event))
    else:
        weight = np.asarray(weight, dtype=float).ravel()

    if encircled_fraction <= 0.0:
        return 0.0
    if encircled_fraction >= 100.0:
        return float(np.max(event))

    total_events = np.sum(weight)
    goal_events  = total_events * encircled_fraction / 100.0

    delta_width       = float(np.max(event))
    half_width_val    = float(np.max(event))
    encircled_events  = total_events
    n_encircled       = len(event)
    i_loop            = 0

    while (abs(encircled_events - goal_events) / goal_events >= goal_precision
           and i_loop < n_loop_max
           and n_encircled >= 0):
        sign = (goal_events - encircled_events) / abs(goal_events - encircled_events)
        delta_width    = abs(delta_width) / 2.0 * sign
        half_width_val = half_width_val + delta_width

        mask = event < half_width_val
        n_encircled = int(np.sum(mask))
        if n_encircled > 0:
            encircled_events = np.sum(weight[mask])
        else:
            encircled_events = 0.0
        i_loop += 1

    if n_encircled <= 0:
        return 999.0

    return half_width_val
