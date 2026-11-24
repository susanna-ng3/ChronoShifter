[README.md](https://github.com/user-attachments/files/35372292/README.md)
# ChronoShifter

**A point cloud transformation tool for modernization of the NSRS**

ChronoShifter is a Python desktop application that transforms LAS/LAZ point clouds from NAD83(2011) to the modernized National Spatial Reference System (NSRS): ITRF2020/NATRF2022 positions projected to the State Plane Coordinate System of 2022 (SPCS2022), with heights referenced to GEOID2022.

It follows the approach that NGS's Horizontal Time-Dependent Positioning (HTDP) tool uses for ECEF coordinates. Positions move between reference frames with a time-dependent 14-parameter Helmert transformation and between epochs with HTDP velocities. ChronoShifter applies that workflow to millions of points at once, with no per-point readjustment.

> **Status:** Research software developed for an M.S. thesis at Oregon State University (2026). SPCS2022 and GEOID2022 are NGS beta products, and zone definitions and grid files may change before the official release. Check results against NGS tools (HTDP, NCAT) before using them in production survey work.

---

## Features

- **Batch LAS/LAZ processing.** Keeps all point attributes (intensity, classification, RGB, extra dimensions).
- **Time-dependent 14-parameter Helmert.** NAD83(2011) → ITRF2020 using HTDP Table 5 parameters and rates (reference epoch 2010.0).
- **Epoch propagation.** You enter the source epoch (e.g., 2010.0) and the target epoch (e.g., 2020.0).
- **Two velocity modes:**
  - *Helmert only*: needs no external files.
  - *HTDP offset correction*: calls `htdp.exe` at the dataset centroid and applies the resulting HTDP − Helmert offset to every point. Four corner points are also run as a diagnostic, and the log reports how much velocity varies across the dataset.
- **Geoid handling.** Removes GEOID18 or GEOID12B to recover ellipsoid heights, then applies GEOID2022 for NAPGD2022 orthometric heights. Reads NGS `.bin` grids directly. GeoTIFF grids work through PROJ/PDAL when available.
- **SPCS2022 projection.** Transverse Mercator (EPSG method 9807), Lambert Conformal Conic 1SP (9801), and Oblique Mercator Conformal / Hotine variant B (9815). Parameters come from the NSRS PROJ auxiliary database.
- **Detailed logging.** Every run writes a `.log` next to its output with sample ECEF coordinates at each step, so you can check them against HTDP and NCAT.

---

## Transformation pipeline

```
NAD83(2011) SPCS / UTM  (E, N, H_NAVD88)  @ source epoch
        │  1. Read LAS/LAZ
        │  2. Unproject → NAD83(2011) geographic          (pyproj)
        │  3. Remove GEOID18 / GEOID12B → ellipsoid height h = H + N
        │  4. Geographic → ECEF (GRS80)
        │  5. 14-parameter Helmert → ITRF2020             (HTDP Table 5)
        │  6. [optional] HTDP offset correction           (htdp.exe)
        │  7. ECEF → ITRF2020 geographic @ target epoch
        │  8. Apply GEOID2022 → orthometric height H = h − N
        │  9. Project → SPCS2022 zone                     (nsrs_proj.db)
        ▼ 10. Write LAS/LAZ + transformation log
NATRF2022 / SPCS2022  (E, N, H_GEOID2022)  @ target epoch
```

---

## Repository structure

| File | Purpose |
|---|---|
| `GUI_transformation_tool.py` | Tkinter GUI and entry point |
| `transformation_core.py` | Main in-memory transformation pipeline and the HTDP subprocess wrapper |
| `transformation_utils.py` | Helmert parameters and functions, PROJ string builders, CRS detection, validation and logging helpers |
| `htdp_velocity.py` | HTDP velocity grid parser (`initvl.f`), NEU↔XYZ velocity conversion, ECEF helpers, Euler-pole plate model |
| `geoid_utils.py` | NGS binary geoid reader (bilinear interpolation) and the GeoTIFF/PDAL geoid corrector |
| `spcs2022_zone_builder.py` | Builds a pyproj CRS for an SPCS2022 zone from the zone-definitions CSV |
| `state_bounds.py` | State lookup from coordinates |
| `zone_code_mapping.py` | Maps SPCS2022 zone codes to NSRS database codes |
| `SPCS2022_Zone_Definitions.csv` | SPCS2022 zone parameters (NGS beta) |
| `SPCS22_Zone_Bounds_beta.csv` | Zone extents used for auto-detection |
| `nsrs_proj.db` | NSRS PROJ auxiliary database (NATRF2022/SPCS2022 definitions) |

---

## Requirements

- Python 3.10 or newer (developed on Python 3.13 / Anaconda, Windows)
- Python packages:

  ```
  numpy
  pandas
  pyproj
  laspy[lazrs]
  Pillow
  ```

- **Optional:**
  - [PDAL](https://pdal.io/) on your `PATH`, for GeoTIFF geoid grids
  - [HTDP](https://geodesy.noaa.gov/TOOLS/Htdp/Htdp.shtml) executable (`htdp.exe`, v3.5.0 or later), for HTDP offset mode

### External data, downloaded separately from NGS

ChronoShifter does not include these files. Download them from NGS:

| Data | Used for |
|---|---|
| GEOID18 binary (e.g., `g2018u0.bin`) | Removing NAVD88 heights |
| GEOID12B binary (optional) | Legacy height removal |
| GEOID2022 binary or GeoTIFF | NAPGD2022 orthometric heights |
| `htdp.exe` | HTDP offset correction mode |

If the geoid files are in the working directory, the GUI detects them automatically. Otherwise, browse to them on the **Geoid** tab.

---

## Installation

```bash
git clone https://github.com/<your-username>/ChronoShifter.git
cd ChronoShifter

conda create -n chronoshifter python=3.13
conda activate chronoshifter
pip install numpy pandas pyproj "laspy[lazrs]" Pillow

# optional
conda install -c conda-forge pdal
```

---

## Usage

```bash
python GUI_transformation_tool.py
```

1. **Input.** Add one or more `.las` or `.laz` files. Enter the input CRS (e.g., `EPSG:6558`) or click **Auto-detect**.
2. **Output.** Pick the state and SPCS2022 zone, or let the tool detect them. Output files get the suffix `_SPCS22_<ZoneAbrv>`.
3. **Epochs.** Enter the source epoch (the epoch of the input data, e.g., `2010.0`) and the target epoch (e.g., `2020.0`).
4. **Velocity method.** Choose *Helmert only*, or *HTDP offset correction* and browse to `htdp.exe`.
5. **Geoid tab.** Select GEOID18 (or GEOID12B) to remove, and GEOID2022 to apply.
6. Click **Transform**. Progress appears on the **Log** tab. A `<output>_transformation.log` file is written next to each output.

### Validating results

Each log lists the sample point's ECEF coordinates in NAD83(2011) at the source epoch and in ITRF2020 at the target epoch. Paste them into [HTDP](https://geodesy.noaa.gov/TOOLS/Htdp/Htdp.shtml) or [NCAT](https://geodesy.noaa.gov/NCAT/) to check the transformation independently.
---

---

## Known limitations

- SPCS2022 zones and GEOID2022 are NGS **beta** releases without official EPSG codes. Zone parameters are read from the bundled CSV and `nsrs_proj.db`.
- Input must be NAD83(2011). Earlier NAD83 realizations are not supported yet.
- The HTDP offset is computed at a single point (the dataset centroid). In areas where velocity varies a lot, check the velocity-uniformity assessment in the log.
- HTDP velocity grids store ITRF2008-frame velocities, and component order differs by region.
- LAS stores coordinates as scaled 32-bit integers, so very large coordinate values lose some precision.
- Developed and tested on Windows. The HTDP integration expects user to have `htdp.exe` installed locally prior to running.
---

## Authors

- **Susanna S. Eng, LSIT**: student developer
- **Dr. Michael J. Olsen**: advisor, Oregon State University

## Acknowledgments

This work was supported by Oregon State University's Geomatics program, Geospatial Center for the Arctic and Pacific (GCAP), the National Geodetic Survey (NGS), the Oregon Department of Transportation (ODOT), and PacTrans. HTDP, NCAT, GEOID models, and SPCS2022 definitions are products of NOAA's National Geodetic Survey.

## Disclaimer

This software is provided as-is for research and educational purposes. It is not an official NGS product. Users are responsible for verifying coordinates before relying on them for surveying, engineering, or legal work.
