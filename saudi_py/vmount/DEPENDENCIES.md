# Virtual-Mounting (PVM) — Dependency Inventory

Full dependency map for the IDL virtual-mounting simulation and its Python port
(`saudi_py/vmount/`).

IDL entry command (the lab's usage):

```idl
LVDTanalysis, module="PVM", vlayer=indgen(1)+4, /HPDsummary, /reloadlaser
```

Input: one laser `.sxd` file. Output: a 6-panel `_Mount.ps`.
IDL source root: `IDL from Jason/user_contrib/saudi/`.

---

## 1. Call graph (mount-simulation path only)

```
LVDTanalysis.pro                         (driver; PVM/reloadlaser/HPDsummary)
└─ LoadLaserSampleFiles.pro  [MakePlots] (the mount physics + _Mount.ps live here)
   ├─ lowpassfilter.pro                  axial low/high-pass split
   ├─ surf_mount.pro                     virtually mount each band
   │   └─ MountDeflection.pro            Euler-beam deflection through spacers (SVD)
   ├─ surf_der.pro                       recompute slopes of mounted surface
   ├─ surf_write.pro                     write <name>_mntsim.sxd
   └─ Plot_LaserMountSim.pro             draw the 2x3 figure
       └─ surf_plot2B.pro                one contour panel + annotation
           └─ surf_HPD2B.pro             analytic 2-bounce HPD
               ├─ surf_der.pro
               └─ half_width.pro
```

The LVDT-text ingestion, laser-vs-LVDT comparison scans (`Surf_CompareScans`),
and the multi-layer `/HPDsummary` summary tables are **out of scope** (not part
of the `.sxd -> _Mount.ps` path being ported).

---

## 2. IDL routine → Python status

### Ported in `saudi_py/vmount/`
| IDL routine (file) | Python module | Notes |
|---|---|---|
| `MountDeflection.pro` | `mountdeflection.py` | SVD beam solve → `np.linalg.lstsq`; force-only + moment variants |
| `surf_HPD2B.pro` | `surf_hpd2b.py` | analytic 2-bounce HPD; `ksp`/`knotsp` spacer masks |
| `lowpassfilter.pro` (+ `Fit_LowPassFilter`) | `lowpassfilter.py` | Butterworth + endpoint fit (`mpfitfun`→`least_squares`) + sigma-weeding |
| `surf_mount.pro` | `surf_mount.py` | default branch (used) + `simple`/`fixed`; `interpol`→scipy, `poly_fit`→numpy |
| `surf_plot2B.pro` (contour mode) | `surf_plot2b.py` | `CONTOUR`→matplotlib; annotation moved off the grid |
| `Plot_LaserMountSim.pro` + `LoadLaserSampleFiles.pro` [MakePlots] | `virtual_mount.py` | orchestrator; hosts the (unseeded) spacer-error RNG; `sp_drerr=` injects a realization (IDL-match validation, see `idl_out/compare_py_idl.py`); `surf_mount` called with `bowfit=True` per IDL `/bowfit` |
| `restore` / `surf_write.pro` (read side) | `sxd_io.py` | reads `.sxd` (XDR) and `.sxd.npz` |
| `BabyIAXO_geometry.txt` read | `load_optics_geometry.py` | per-layer radius/angle; feeds only the unused `dr_conic` |
| — (CLI) | `run_virtual_mount.py` | repo-root runner |

### Reused from the existing `saudi_py/` port
| IDL routine | Python | 
|---|---|
| `surf_der.pro` | `saudi_py/surf_der.py` |
| `half_width.pro` | `saudi_py/half_width.py` |
| `surf_write.pro` (write side) | `saudi_py/surf_write.py` |

### External libraries → Python replacements
| IDL dependency | Origin | Python replacement |
|---|---|---|
| `mpfit`, `mpfitfun` | Markwardt `cmtotal/` | `scipy.optimize.least_squares` |
| `readcol` | IDL AstroLib `astron/` | `numpy.loadtxt` |
| `interpol` (`/quad`, `/spline`) | IDL built-in | `scipy.interpolate.interp1d` / `CubicSpline` |
| `deriv` | IDL built-in | `numpy.gradient` (via `surf_der`) |
| `randomu` (`/normal`) | IDL built-in | `numpy.random.default_rng` (unseeded by default) |
| `SVDC` / `SVSOL` | IDL built-in | `numpy.linalg.lstsq` |
| `smooth` | IDL built-in | boxcar convolution (ends unchanged) |
| `fft` | IDL built-in | `numpy.fft` (IDL sign/normalization reproduced) |
| `CONTOUR`, `SHADE_SURF`, PostScript device | IDL graphics | `matplotlib` (contour → PDF/PNG) |
| `plot_text` | windt/ | inline text in a reserved axes |

### Not required by this path (present in the full driver only)
`Surf_CompareScans`, `surf_2b`/`surf_2b_trace` (full MC raytrace), `surf_2b_image`,
`HPDfunction`, `newwindow`, `sformat`, `XRC_MakePageTitle`, `legend`, `multiplot`,
Coyote graphics (`cgtext`, …), `LoadOpticsGeometryNuSTAR` (NuSTAR reference geometry).

---

## 3. Data files
- **`BabyIAXO_geometry.txt`** — per-layer radius/cone-angle table. **Not present** in the
  IDL tree (lives in the lab `scan/SXD/`). Only feeds `dr_conic`, which is computed but
  **unused** downstream, so absence does not affect the `_Mount` output. `--geometry`
  supplies it if available.
- **`NuSTAR_PVM_Mounted.txt`** — SN↔mount-position lookup, used only by the out-of-scope
  driver (`/reloadlaser` batch); not needed for a single-file mount run.

---

## 4. Duplicate / ambiguous IDL files (resolved to canonical)
The IDL tree contains many `Copy of …`, `… - Copy (N)`, `_old`, `.BAK`, and
`.svn/text-base/*.svn-base` variants. The port uses the canonical non-`Copy`/non-`_old`
file in `saudi/`. `vector()` has no definition (only windt `MakeVector`); it is a simple
linspace helper, reimplemented inline where needed.

---

## 5. Python package requirements
`numpy`, `scipy` (`optimize`, `interpolate`, `io.readsav`), `matplotlib`.
(Same stack as the existing `saudi_py` port — no new third-party dependencies.)
