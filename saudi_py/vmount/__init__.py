"""
saudi_py.vmount — Virtual-mounting (PVM) simulation.

Python port of the IDL laser-scanner *virtual mounting* analysis
(`IDL from Jason/user_contrib/saudi/`): given a laser-scanned mirror shell
(`.sxd`), simulate how the free-standing glass elastically deforms once
epoxied onto graphite spacers, and report the resulting 2-bounce HPD.

The IDL entry point is

    LVDTanalysis, module="PVM", vlayer=indgen(1)+N, /HPDsummary, /reloadlaser

whose mount physics lives in LoadLaserSampleFiles.pro (the MakePlots block):
    read .sxd -> surf_mount -> MountDeflection -> Plot_LaserMountSim
                                              -> surf_plot2b / surf_HPD2b

This subpackage ports that core chain only (the LVDT-text ingestion, the
laser-vs-LVDT comparison scans, and the multi-layer /HPDsummary tables are
out of scope).

Modules
-------
sxd_io               read a shell surface from .sxd (IDL XDR) or .sxd.npz
mountdeflection      Euler-Bernoulli beam deflection through spacer constraints
surf_hpd2b           analytic 2-bounce HPD from a surface
lowpassfilter        LVDT-path Butterworth low-pass filter (endpoint-fitted)
surf_mount           virtually mount a laser surface onto spacers
surf_plot2b          one contour panel with HPD annotation
load_optics_geometry per-layer radius / cone-angle lookup (BabyIAXO_geometry.txt)
virtual_mount        orchestrator: .sxd -> 6-panel mount figure + HPDs
"""

from .virtual_mount import virtual_mount

__all__ = ["virtual_mount"]
