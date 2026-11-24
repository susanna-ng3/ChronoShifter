"""
CORRECTED HTDP Velocity Grid Module
Applies crustal velocity corrections using HTDP methodology

CRITICAL FIX: HTDP applies velocities in XYZ space, NOT geodetic/NEU space!

Workflow:
1. Get velocity from grid(s) (NEU, mm/yr) - supports overlapping grid SUM
2. Convert NEU velocity to XYZ velocity
3. Apply XYZ displacement in Cartesian space
4. Transform frame using Helmert (if needed)

Authors: Susanna Eng, Michael Olsen
Date: November 2025
"""

import numpy as np
import re
import math
import logging
import os
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def get_resource_path(filename):
    """
    Resolve resource file path for both development and PyInstaller --onefile.
    PyInstaller extracts data to sys._MEIPASS in --onefile mode.
    """
    paths_to_check = []

    # 1. PyInstaller --onefile temporary directory
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        meipass = sys._MEIPASS
        candidate = os.path.join(meipass, filename)
        paths_to_check.append(candidate)
        logging.debug(f"Checking PyInstaller path for {filename}: {candidate}")
        if os.path.exists(candidate):
            logging.info(f"Found resource at: {candidate}")
            return candidate

    # 2. Current working directory
    cwd_candidate = os.path.join(os.getcwd(), filename)
    paths_to_check.append(cwd_candidate)
    logging.debug(f"Checking cwd path for {filename}: {cwd_candidate}")
    if os.path.exists(cwd_candidate):
        logging.info(f"Found resource at: {cwd_candidate}")
        return cwd_candidate

    # 3. Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_candidate = os.path.join(script_dir, filename)
    paths_to_check.append(script_candidate)
    logging.debug(f"Checking script path for {filename}: {script_candidate}")
    if os.path.exists(script_candidate):
        logging.info(f"Found resource at: {script_candidate}")
        return script_candidate

    # Not found - log all attempted paths
    logging.error(f"Resource '{filename}' not found. Searched:")
    for path in paths_to_check:
        logging.error(f"  - {path}")

    # Return first preference even if not found (will error at open())
    return paths_to_check[0]


def _lon_positive_west(lon_deg: float) -> float:
    return lon_deg + 360.0 if lon_deg < 0.0 else lon_deg


import csv
import math
import numpy as np
from dataclasses import dataclass


# --- replace your PlateModelDB with this ---

@dataclass
class PlateEuler:
    name: str
    pole_lat_deg: float
    pole_lon_deg: float
    omega_deg_per_myr: float  # rotation rate


@dataclass
class PlateRateVector:
    name: str
    rx_rad_per_yr: float
    ry_rad_per_yr: float
    rz_rad_per_yr: float
    tx_mm_per_yr: float = 0.0
    ty_mm_per_yr: float = 0.0
    tz_mm_per_yr: float = 0.0


class PlateModelDB:
    def __init__(self, csv_path: str, default_plate: str = "NorthAmerica"):
        import csv, re, math
        self._eulers: dict[str, PlateEuler] = {}
        self._rates: dict[str, PlateRateVector] = {}

        def canon(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            cols = [c for c in rdr.fieldnames or []]
            ccols = [canon(c) for c in cols]

            # Detect schema
            has_euler = {"polelat", "polelatdeg", "lat", "latdeg"} & set(ccols)
            has_rate = {"rxnradperyr", "rynradperyr", "rznradperyr", "rx_rad_per_yr", "ry_rad_per_yr",
                        "rz_rad_per_yr"} & set(ccols)

            for row in rdr:
                rowc = {canon(k): v for k, v in row.items()}
                # plate name
                name = row.get("PlateName") or row.get("Plate") or row.get("plate") or row.get("name") or "UnnamedPlate"

                if has_rate:
                    # Schema B: angular rates (nrad/yr) and optional translational rates (mm/yr)
                    def g(key_alts, default="0"):
                        for k in key_alts:
                            if canon(k) in rowc:
                                return row[rowc.keys().__iter__().__next__()]  # not used
                        return default  # (we will fetch by canonical below)

                    rx = float(rowc.get("rxnradperyr", rowc.get("rx_rad_per_yr", "0"))) * 1e-9  # nrad/yr -> rad/yr
                    ry = float(rowc.get("rynradperyr", rowc.get("ry_rad_per_yr", "0"))) * 1e-9
                    rz = float(rowc.get("rznradperyr", rowc.get("rz_rad_per_yr", "0"))) * 1e-9
                    tx = float(rowc.get("txmmperyr", "0"))
                    ty = float(rowc.get("tymmperyr", "0"))
                    tz = float(rowc.get("tzmmperyr", "0"))
                    self._rates[name] = PlateRateVector(name, rx, ry, rz, tx, ty, tz)
                elif has_euler:
                    # Schema A: Euler pole (deg, deg, deg/Myr)
                    # Try a few header variants
                    def fetch(keys):
                        for k in keys:
                            if canon(k) in rowc:
                                return float(row[row.keys().__iter__().__next__()])  # not used
                        return None

                    # Use canonical dict directly:
                    def f(key, default=None):
                        v = rowc.get(key, None)
                        return float(v) if (v is not None and v != "") else default

                    plat = f("polelatdeg", f("polelat", f("latdeg", f("lat"))))
                    plon = f("polelondeg", f("polelon", f("londeg", f("lon"))))
                    # angular rate: prefer deg/Myr; also accept deg/yr or rad/yr
                    if "omegadegpermyr" in rowc or "omegadegmyr" in rowc or "omegadeg/ma" in rowc:
                        omg_deg_myr = float(
                            rowc.get("omegadegpermyr", rowc.get("omegadegmyr", rowc.get("omegadeg/ma"))))
                    elif "omegadegperyr" in rowc or "omegadeg/yr" in rowc:
                        omg_deg_myr = float(rowc.get("omegadegperyr", rowc.get("omegadeg/yr"))) * 1e6
                    elif "omegaradperyr" in rowc or "omegarad/yr" in rowc:
                        omg_deg_myr = math.degrees(float(rowc.get("omegaradperyr", rowc.get("omegarad/yr")))) * 1e6
                    else:
                        raise KeyError("Euler-pole table missing omega column (deg/Myr, deg/yr, or rad/yr).")
                    if plat is None or plon is None:
                        raise KeyError("Euler-pole table missing pole latitude/longitude columns.")
                    self._eulers[name] = PlateEuler(name, plat, plon, omg_deg_myr)
                else:
                    raise KeyError(
                        "Unrecognized plate model schema. Found columns: "
                        + ", ".join(cols) +
                        ". Need either Euler-pole columns (pole_lat, pole_lon, omega) or "
                        "rate-vector columns (Rx_nrad_per_yr, Ry_nrad_per_yr, Rz_nrad_per_yr)."
                    )

        # Choose default plate
        all_names = list(self._eulers.keys() | self._rates.keys())
        if default_plate not in all_names:
            for k in all_names:
                if "north" in k.lower() and "amer" in k.lower():
                    default_plate = k
                    break
            else:
                default_plate = all_names[0] if all_names else default_plate
        self.default_plate = default_plate

        # Small log
        try:
            if self.default_plate in self._rates:
                r = self._rates[self.default_plate]
                print(f"[PlateModelDB] Loaded {len(all_names)} plates (rate-vector). Default='{self.default_plate}' "
                      f"(ω=({r.rx_rad_per_yr:.3e},{r.ry_rad_per_yr:.3e},{r.rz_rad_per_yr:.3e}) rad/yr).")
            else:
                e = self._eulers[self.default_plate]
                print(f"[PlateModelDB] Loaded {len(all_names)} plates (Euler). Default='{self.default_plate}' "
                      f"(pole {e.pole_lat_deg:.2f}deg, {e.pole_lon_deg:.2f}deg, ω={e.omega_deg_per_myr:.3f} deg/Myr).")
        except Exception:
            pass

    def velocity_neu_mm_per_yr(self, lat_deg: float, lon_deg: float, plate_name: str | None = None):
        import numpy as np, math
        name = plate_name or self.default_plate
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)

        # Site unit vector
        sx = math.cos(lat) * math.cos(lon)
        sy = math.cos(lat) * math.sin(lon)
        sz = math.sin(lat)
        r_hat = np.array([sx, sy, sz], dtype=np.float64)

        # Local ENU rotation
        slat, clat = math.sin(lat), math.cos(lat)
        slon, clon = math.sin(lon), math.cos(lon)
        R_T = np.array([
            [-slat * clon, -slat * slon, clat],  # N
            [-slon, clon, 0.0],  # E
            [clat * clon, clat * slon, slat],  # U
        ], dtype=np.float64)

        R_earth_m = 6371000.0

        if name in self._rates:
            # Schema B: direct angular rate vector (rad/yr) and optional translation (mm/yr in ECEF)
            r = self._rates[name]
            w = np.array([r.rx_rad_per_yr, r.ry_rad_per_yr, r.rz_rad_per_yr], dtype=np.float64)  # rad/yr
            v_ecef_m_per_yr = np.cross(w, R_earth_m * r_hat)
            v_ecef_mm_per_yr = 1000.0 * v_ecef_m_per_yr + np.array(
                [r.tx_mm_per_yr, r.ty_mm_per_yr, getattr(r, "tz_mm_per_yr", 0.0)])
            vN, vE, vU = (R_T @ v_ecef_mm_per_yr).tolist()
            # Rigid plate ⇒ no vertical from rotation; translation may add tiny U; keep as-is
            return float(vN), float(vE), float(vU)

        # Schema A: Euler pole (deg/Myr)
        e = self._eulers[name]
        omega_rad_per_yr = math.radians(e.omega_deg_per_myr) / 1e6
        ux = math.cos(math.radians(e.pole_lat_deg)) * math.cos(math.radians(e.pole_lon_deg))
        uy = math.cos(math.radians(e.pole_lat_deg)) * math.sin(math.radians(e.pole_lon_deg))
        uz = math.sin(math.radians(e.pole_lat_deg))
        w = np.array([ux, uy, uz], dtype=np.float64) * omega_rad_per_yr  # rad/yr

        v_ecef_m_per_yr = np.cross(w, R_earth_m * r_hat)
        vN, vE, vU = (R_T @ (1000.0 * v_ecef_m_per_yr)).tolist()
        return float(vN), float(vE), 0.0  # Euler rigid plate: U~=0


# =========================
# Grid container
# =========================
@dataclass
class VelocityGrid:
    """HTDP velocity grid with selectable interpolation (NEU, mm/yr).

    Supports: bilinear (fast), bicubic (smooth, slower)
    """
    region_number: int
    region_name: str
    lat_min: float
    lat_max: float
    lon_min: float  # NOTE: HTDP longitudes are 0–360deg, positive west
    lon_max: float
    spacing_arcmin: float
    velocities: np.ndarray  # shape (n_lat, n_lon, 3) ordered as [N, E, U] in mm/yr
    start_index: int
    interpolation_method: str = 'bilinear'  # NEW: selectable method

    def is_point_in_grid(self, lat: float, lon: float) -> bool:
        """Check if (lat, lon) lies within grid bounds; HTDP uses 0–360deg (positive west) for longitudes."""
        lon_pw = (360.0 + lon) if lon < 0 else lon  # convert to 0–360 if lon is negative (west)
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon_pw <= self.lon_max)

    def _bilinear_indices_weights(lat, lon, lat_min, lon_min, ddeg, n_lat, n_lon):
        """Return grid indices (i0,j0) and bilinear weights (w00,w10,w01,w11)."""
        # fractional indices
        fi = (lat - lat_min) / ddeg
        fj = (lon - lon_min) / ddeg
        i0 = int(math.floor(fi))
        j0 = int(math.floor(fj))
        # clamp to interior
        i0 = max(0, min(n_lat - 2, i0))
        j0 = max(0, min(n_lon - 2, j0))
        di = fi - i0
        dj = fj - j0
        w00 = (1 - di) * (1 - dj)
        w10 = di * (1 - dj)
        w01 = (1 - di) * dj
        w11 = di * dj
        return i0, j0, w00, w10, w01, w11

    def get_velocity_at_point(self, lat_deg: float, lon_deg: float):
        """Get velocity at a specific point using selected interpolation method.

        Args:
            lat_deg, lon_deg: Geographic coordinates (negative lon = west)

        Returns:
            (vN, vE, vU) in mm/yr, or None if outside grid
        """

        # Convert longitude if needed
        lon_htdp = lon_deg + 360 if lon_deg < 0 else lon_deg

        # Check bounds
        if not (self.lat_min <= lat_deg <= self.lat_max and
                self.lon_min <= lon_htdp <= self.lon_max):
            return None

        # ADAPTIVE STRATEGY: Use cell-center sampling for coarse grids
        use_cell_centers = self.spacing_arcmin >= 10.0

        # Calculate fractional grid indices
        lat_frac = (lat_deg - self.lat_min) * 60.0 / self.spacing_arcmin
        lon_frac = (lon_htdp - self.lon_min) * 60.0 / self.spacing_arcmin

        # Shift by 0.5 cells if using cell centers
        if use_cell_centers:
            lat_frac -= 0.5
            lon_frac -= 0.5

        lat_idx = int(lat_frac)
        lon_idx = int(lon_frac)

        # Check we can interpolate
        if (lat_idx < 0 or lat_idx >= self.velocities.shape[0] - 1 or
                lon_idx < 0 or lon_idx >= self.velocities.shape[1] - 1):
            return None

        # Fractional parts
        wlat = lat_frac - lat_idx
        wlon = lon_frac - lon_idx

        # Dispatch to selected interpolation method
        if self.interpolation_method == 'bicubic':
            return self._interpolate_bicubic(lat_idx, lon_idx, wlat, wlon)
        else:  # bilinear (default)
            return self._interpolate_bilinear(lat_idx, lon_idx, wlat, wlon)

    def _interpolate_bilinear(self, lat_idx, lon_idx, wlat, wlon):
        """Bilinear interpolation (current default, fast)."""
        v00 = self.velocities[lat_idx, lon_idx]
        v01 = self.velocities[lat_idx, lon_idx + 1]
        v10 = self.velocities[lat_idx + 1, lon_idx]
        v11 = self.velocities[lat_idx + 1, lon_idx + 1]

        v = ((1 - wlat) * (1 - wlon) * v00 +
             (1 - wlat) * wlon * v01 +
             wlat * (1 - wlon) * v10 +
             wlat * wlon * v11)

        return tuple(v)

    def _interpolate_bicubic(self, lat_idx, lon_idx, wlat, wlon):
        """Bicubic interpolation using Catmull-Rom basis (smooth, +/-1-2mm accuracy)."""
        import numpy as np

        # Catmull-Rom basis functions
        def catmull_rom(t):
            """Catmull-Rom basis function."""
            t_sq = t * t
            t_cu = t_sq * t
            return np.array([
                -0.5 * t_cu + t_sq - 0.5 * t,  # h00
                1.5 * t_cu - 2.5 * t_sq + 1.0,  # h10
                -1.5 * t_cu + 2.0 * t_sq + 0.5 * t,  # h01
                0.5 * t_cu - 0.5 * t_sq  # h11
            ])

        # Get basis functions for both directions
        h_lat = catmull_rom(wlat)
        h_lon = catmull_rom(wlon)

        # Sample 4x4 grid with clamping at boundaries
        v = np.zeros(3)
        for i in range(4):
            for j in range(4):
                di = lat_idx - 1 + i
                dj = lon_idx - 1 + j

                # Clamp to valid indices
                di = np.clip(di, 0, self.velocities.shape[0] - 1)
                dj = np.clip(dj, 0, self.velocities.shape[1] - 1)

                v += h_lat[i] * h_lon[j] * self.velocities[di, dj]

        return tuple(v)


# =========================
# Grids manager (overlap SUM)
# =========================
class HTDPVelocityGrids:
    """
    Manager for HTDP velocity grids with optional, rate-limited diagnostics and overlap control.

    overlap_mode:
      - "sum" (default): add NEU contributions from ALL grids that include the point
                         (matches HTDP behavior in regions with overlapping grids)
      - "highest": use only the highest-resolution grid that includes the point
    """

    def __new__(
            cls,
            initvl_path: Optional[str] = None,
            verbose: bool = False,
            log_once: bool = True,
            max_logs_per_run: int = 1,
            overlap_mode: str = "sum",
            vertical_override_mm_per_yr: Optional[float] = None,
    ):
        # Singleton pattern REMOVED - always create new instance
        return super().__new__(cls)

    def __init__(
            self,
            initvl_path: Optional[str] = None,
            verbose: bool = False,
            log_once: bool = True,
            max_logs_per_run: int = 1,
            overlap_mode: str = "sum",
            vertical_override_mm_per_yr: Optional[float] = None,
    ):
        # Only initialize once when we actually build the grids
        if not hasattr(self, 'grids') and initvl_path is not None:
            self._verbose = bool(verbose)
            self._log_once = bool(log_once)
            self._max_logs = int(max(0, max_logs_per_run))
            self._grid_log_count = 0
            om = (overlap_mode or "highest").lower().strip()
            self._overlap_mode = om if om in ("sum", "highest") else "highest"

            # NEW: store the optional vertical override (mm/yr)
            self._v_override = None if vertical_override_mm_per_yr is None else float(vertical_override_mm_per_yr)

            self.grids = self._parse_initvl(initvl_path)
            logger.info(
                f"HTDP velocity grids loaded: {len(self.grids)} grids "
                f"(overlap_mode='{self._overlap_mode}')"
            )
            # Add after parsing grids
            self.plate_db = PlateModelDB(get_resource_path("plateModels.csv"), default_plate="NorthAmerica")
            self.use_plate_blend = True
            self.seam_blend = True
            self.use_plate_blend = True  # fallback only (code above enforces that)
            self.sample_cell_center = True

            for g in self.grids:
                setattr(g, "sample_cell_center", True)

    def set_vertical_override(self, value: Optional[float]):
        """Enable/disable a global vertical-rate override (mm/yr)."""
        self._v_override = None if value is None else float(value)

    def reset_grid_logging(self):
        """Reset the per-run diagnostic counter; call this between jobs/runs."""
        self._grid_log_count = 0

    # ----- parsing -----
    def _parse_initvl(self, filepath: str) -> List[VelocityGrid]:
        """Parse initvl.f and extract velocity grids into NEU arrays (mm/yr)."""
        import os
        logger.info(f"Loading HTDP velocity grids from {filepath}")

        # Check if file exists and has content
        if not os.path.exists(filepath):
            raise ValueError(f"HTDP velocity file not found: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise ValueError(f"HTDP velocity file is empty: {filepath}")

        logger.info(f"File size: {file_size:,} bytes")

        text = Path(filepath).read_text(encoding='utf-8', errors='ignore')

        # Count DATA B lines for diagnostics
        data_b_lines = text.count('DATA B(')
        logger.info(f"Found {data_b_lines} 'DATA B(' lines in file")

        # ---- 1) Parse the Fortran DATA B(...) blocks into a dense 1-based array ----
        # Matches lines like:  DATA B(    1), B(   2), ... /  -0.23,  0.15,  0.02, ... /
        pattern = r'DATA\s+B\(\s*(\d+)\s*\)(?:,\s*B\(\s*\d+\s*\))*\s*/\s*([^/]+)\s*/'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        logger.info(f"Regex matched {len(matches)} DATA B(...) blocks")

        b_values = {}
        for m in matches:
            first_idx = int(m.group(1))
            vals = [v.strip() for v in m.group(2).split(',') if v.strip()]
            for i, v in enumerate(vals):
                # Fortran arrays are 1-based; keep that indexing here
                b_values[first_idx + i] = float(v)

        if not b_values:
            raise ValueError(
                f"No velocity values found in {filepath}.\n"
                f"File details: size={file_size:,} bytes, "
                f"'DATA B(' occurrences={data_b_lines}, "
                f"regex matches={len(matches)}.\n"
                f"Make sure you're using the correct initvl.f file format."
            )

        max_index = max(b_values.keys())
        # Dense 1-based backing array so that b_array[k] is valid for k>=1
        b_array = np.zeros(max_index + 1, dtype=float)
        for idx, val in b_values.items():
            b_array[idx] = val

        logger.info(f"Parsed {len(b_values)} B array values "
                    f"(B[1]={b_array[1]:.3f}, B[2]={b_array[2]:.3f}, B[3]={b_array[3]:.3f})")

        # ---- 2) Region metadata (HTDP longitudes: 0–360deg, positive west) ----
        grid_params = [
            # CONUS regions (NBASE from initvl.f)
            {'region': 1, 'name': 'San Andreas',
             'lat_min': 35.8, 'lat_max': 36.79, 'lon_min': 238.2, 'lon_max': 239.49, 'spacing': 0.60,
             'n_lat': 99, 'n_lon': 129, 'nbase': 0},
            {'region': 2, 'name': 'Southern California',
             'lat_min': 31.0, 'lat_max': 36.0, 'lon_min': 239.0, 'lon_max': 246.0, 'spacing': 3.75,
             'n_lat': 81, 'n_lon': 113, 'nbase': 39000},
            {'region': 3, 'name': 'Northern California',
             'lat_min': 36.0, 'lat_max': 40.0, 'lon_min': 235.0, 'lon_max': 241.0, 'spacing': 3.75,
             'n_lat': 65, 'n_lon': 97, 'nbase': 66459},
            {'region': 4, 'name': 'Pacific Northwest',
             'lat_min': 40.0, 'lat_max': 49.0, 'lon_min': 235.0, 'lon_max': 238.0, 'spacing': 3.75,
             'n_lat': 145, 'n_lon': 49, 'nbase': 85374},
            {'region': 5, 'name': 'Western CONUS',
             'lat_min': 31.0, 'lat_max': 49.0, 'lon_min': 235.0, 'lon_max': 253.0, 'spacing': 15.0,
             'n_lat': 73, 'n_lon': 73, 'nbase': 106689},
            {'region': 6, 'name': 'Eastern CONUS',
             'lat_min': 24.0, 'lat_max': 50.0, 'lon_min': 250.0, 'lon_max': 294.0, 'spacing': 30.0,
             'n_lat': 53, 'n_lon': 89, 'nbase': 128808},
            # ALASKA regions (from initvl.f, correctly defined)
            {'region': 7, 'name': 'St. Elias, Alaska',
             'lat_min': 56.5, 'lat_max': 63.0, 'lon_min': 210.0, 'lon_max': 220.0, 'spacing': 15.0,
             'n_lat': 26, 'n_lon': 40, 'nbase': 142959},
            {'region': 8, 'name': 'South-central Alaska',
             'lat_min': 53.25, 'lat_max': 65.75, 'lon_min': 198.0, 'lon_max': 216.75, 'spacing': 15.0,
             'n_lat': 50, 'n_lon': 75, 'nbase': 146280},
            {'region': 9, 'name': 'Southeast Alaska',
             'lat_min': 54.0, 'lat_max': 64.0, 'lon_min': 218.0, 'lon_max': 230.0, 'spacing': 15.0,
             'n_lat': 40, 'n_lon': 48, 'nbase': 157908},
            {'region': 10, 'name': 'All mainland Alaska',
             'lat_min': 56.0, 'lat_max': 73.0, 'lon_min': 190.0, 'lon_max': 230.0, 'spacing': 15.0,
             'n_lat': 68, 'n_lon': 160, 'nbase': 163935},
        ]

        # ---- 3) PASS 1 - compute start_index for ALL regions first ----
        for p in grid_params:
            n_lat = p['n_lat']
            n_lon = p['n_lon']
            p['start_index'] = p.get('nbase', p.get('start_index', 0))
            logger.info(f"Region {p['region']:d}: {n_lat}x{n_lon} nodes, "
                        f"start B[{p['start_index']}] (spacing {p['spacing']:.2f} arcmin)")

        # ---- 4) PASS 2 - build velocity cubes for each region using the now-known starts ----
        grids: List[VelocityGrid] = []
        for p in grid_params:
            n_lat = p['n_lat']
            n_lon = p['n_lon']
            total_nodes = n_lat * n_lon
            start_idx = p['start_index']

            V = np.zeros((n_lat, n_lon, 3), dtype=float)  # [N, E, U] in mm/yr

            # Fill cell-by-cell from 1D B array
            # NOTE: Most grids use [N, E, U] order, but Pacific Northwest (region 4) uses [U, N, E]
            for node_idx in range(total_nodes):
                i_lat = node_idx // n_lon
                i_lon = node_idx % n_lon
                b_idx = start_idx + node_idx * 3 + 1  # FIXED: Fortran data starts at NBASE+1

                # Safety check: stay inside b_array
                if b_idx + 2 > max_index:
                    # Incomplete tail; leave zeros for this and following nodes
                    break

                # Pacific Northwest (region 4) uses different component ordering
                if p['region'] == 4:
                    # Region 4 stores as [U, N, E]
                    V[i_lat, i_lon, 0] = b_array[b_idx + 1]  # N (from index 1)
                    V[i_lat, i_lon, 1] = b_array[b_idx + 2]  # E (from index 2)
                    V[i_lat, i_lon, 2] = b_array[b_idx + 0]  # U (from index 0)
                else:
                    # Standard [N, E, U] ordering
                    V[i_lat, i_lon, 0] = b_array[b_idx + 0]  # N
                    V[i_lat, i_lon, 1] = b_array[b_idx + 1]  # E
                    V[i_lat, i_lon, 2] = b_array[b_idx + 2]  # U

            grids.append(VelocityGrid(
                region_number=p['region'],
                region_name=p['name'],
                lat_min=p['lat_min'],
                lat_max=p['lat_max'],
                lon_min=p['lon_min'],
                lon_max=p['lon_max'],
                spacing_arcmin=p['spacing'],
                velocities=V,
                start_index=start_idx
            ))

        return grids

    # ----- logging helper -----
    def _maybe_log_selection(self, grid: VelocityGrid, v: Tuple[float, float, float], level_debug_ok: bool):
        if not self._verbose:
            return
        if self._log_once:
            if self._grid_log_count < self._max_logs:
                self._grid_log_count += 1
                logger.info(
                    f"GRID SELECTION: {grid.region_number} ({grid.region_name}), "
                    f"{grid.spacing_arcmin} arcmin - "
                    f"N={v[0]:+.2f}, E={v[1]:+.2f}, Up={v[2]:+.2f} mm/yr"
                )
        elif level_debug_ok and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"GRID SELECTION: {grid.region_number} ({grid.region_name}), "
                f"{grid.spacing_arcmin} arcmin - "
                f"N={v[0]:+.2f}, E={v[1]:+.2f}, Up={v[2]:+.2f} mm/yr"
            )

    # SIMPLE FIX: Apply ITRF2008->ITRF2020 frame transformation in get_velocity()

    def get_velocity(self, lat_deg: float, lon_deg: float, apply_frame_transform=True):
        """
        Return NEU velocity (mm/yr) at (lat, lon).

        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            apply_frame_transform: If True, return velocity in ITRF2020 frame (for ITRF targets)
                                  If False, return raw grid velocity in NAD83 frame (for SPCS2022)
        """
        import numpy as np

        if not apply_frame_transform:
            # Return raw grid velocity without frame transformation
            lon_pw = lon_deg + 360.0 if lon_deg < 0.0 else lon_deg
            candidates = []
            for g in getattr(self, "grids", []) or []:
                v = g.get_velocity_at_point(lat_deg, lon_pw)
                if v is not None:
                    candidates.append((g, v))

            if not candidates:
                if getattr(self, "use_plate_blend", False) and hasattr(self, "plate_db"):
                    return self.plate_db.velocity_neu_mm_per_yr(lat_deg, lon_deg)
                return (0.0, 0.0, 0.0)

            candidates.sort(key=lambda x: x[0].spacing_arcmin)
            g_best, v_best = candidates[0]
            vN, vE, vU = v_best
            return (float(vN), float(vE), float(vU))

        # Convert negative Western longitude to positive-west (0-360)
        lon_pw = lon_deg + 360.0 if lon_deg < 0.0 else lon_deg

        # Find all grids covering this point
        candidates = []
        for g in getattr(self, "grids", []) or []:
            v = g.get_velocity_at_point(lat_deg, lon_pw)
            if v is not None:
                candidates.append((g, v))

        if not candidates:
            if getattr(self, "use_plate_blend", False) and hasattr(self, "plate_db"):
                return self.plate_db.velocity_neu_mm_per_yr(lat_deg, lon_deg)
            return (0.0, 0.0, 0.0)

        # Prefer highest resolution (smallest spacing)
        candidates.sort(key=lambda x: x[0].spacing_arcmin)
        g_best, v_best = candidates[0]

        # Optional seam averaging
        if getattr(self, "seam_blend", False) and len(candidates) > 1:
            g2, v2 = candidates[1]
            if g2.spacing_arcmin <= 2.0 * g_best.spacing_arcmin:
                v_best = tuple((np.array(v_best, dtype=float) + np.array(v2, dtype=float)) * 0.5)

        # Return grid velocities directly
        # TODO: Grid velocities should be in NAD83 frame and need plate transformation
        # But plate model is currently giving wrong values, so using raw grid for now
        vN, vE, vU = v_best
        return (float(vN), float(vE), float(vU))

    def set_htdp_strict(self):
        """Pin knobs to HTDP-like behavior."""
        self._overlap_mode = "hierarchical"  # prefer finest grid
        self._v_override = None  # no vertical override

    # In your HTDPVelocityGrids class
    # Current mode: "highest" - might be selecting wrong grid

    # Try these diagnostic steps:
    def diagnose_grid_coverage(self, lat, lon):
        """Debug which grids contribute at a location"""
        contributing_grids = []
        for grid in self.grids:
            if grid.contains(lat, lon):
                v_n, v_e, v_u = grid.interpolate(lat, lon)
                contributing_grids.append({
                    'name': grid.name,
                    'bounds': grid.bounds,
                    'velocity': (v_n, v_e, v_u)
                })
        return contributing_grids


# =========================
# Geodetic / velocity math
# =========================
def vtranf_iers_itrf2008_to_itrf2020(X, Y, Z, VX, VY, VZ, epoch=2020.0):
    """
    Transform velocities from ITRF2008 to ITRF2020 using VELOCITY RATE parameters.

    CRITICAL FIX: Use velocity rate parameters, not position parameters.
    The IERS transformation specifies velocity rates (dRX/dt, dRY/dt, dRZ/dt, dS/dt)
    which are epoch-INDEPENDENT and apply to all epochs.

    Args:
        X, Y, Z: Position in meters (ECEF)
        VX, VY, VZ: Velocity in mm/yr (ITRF2008 frame)
        epoch: Reference epoch (unused - rates are constant)

    Returns:
        VX, VY, VZ in ITRF2020 (mm/yr)
    """

    def mas_to_radians(mas):
        return (mas / 3600.0) * (math.pi / 180.0)

    # ITRF2008 → ITRF94: Using velocity RATES from IERS Conventions 2010/2020
    # These are time-derivatives of the position transformation parameters
    dRX_08_mas_yr = -0.01347  # mas/yr (constant)
    dRY_08_mas_yr = +0.01514  # mas/yr (constant)
    dRZ_08_mas_yr = +0.01973  # mas/yr (constant)
    dS_08_ppb_yr = +0.10201  # ppb/yr (constant)

    # ITRF94 → ITRF2020: Using velocity RATES from IERS Conventions 2020
    dRX_20_mas_yr = -0.02  # mas/yr (constant, approximate)
    dRY_20_mas_yr = +0.02  # mas/yr (constant, approximate)
    dRZ_20_mas_yr = +0.02  # mas/yr (constant, approximate)
    dS_20_ppb_yr = -0.12  # ppb/yr (constant, approximate)

    # Convert to radians/yr
    dRX_08_rad_yr = mas_to_radians(dRX_08_mas_yr)
    dRY_08_rad_yr = mas_to_radians(dRY_08_mas_yr)
    dRZ_08_rad_yr = mas_to_radians(dRZ_08_mas_yr)
    dS_08 = dS_08_ppb_yr * 1e-9

    dRX_20_rad_yr = mas_to_radians(dRX_20_mas_yr)
    dRY_20_rad_yr = mas_to_radians(dRY_20_mas_yr)
    dRZ_20_rad_yr = mas_to_radians(dRZ_20_mas_yr)
    dS_20 = dS_20_ppb_yr * 1e-9

    # Step 1: Apply ITRF2008→ITRF94 velocity transformation
    # v_B = v_A + dT_rate + (Ω_rate × X) + dS_rate * X
    dT_08 = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # no translation rate
    w_08 = np.array([dRX_08_rad_yr, dRY_08_rad_yr, dRZ_08_rad_yr], dtype=np.float64)
    X_pos = np.array([X, Y, Z], dtype=np.float64)
    V_08 = np.array([VX, VY, VZ], dtype=np.float64)

    cross_08 = np.cross(w_08, X_pos)
    scale_08 = dS_08 * X_pos
    V_94 = V_08 + dT_08 + cross_08 + scale_08

    # Step 2: Apply ITRF94→ITRF2020 velocity transformation
    dT_20 = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # no translation rate
    w_20 = np.array([dRX_20_rad_yr, dRY_20_rad_yr, dRZ_20_rad_yr], dtype=np.float64)

    cross_20 = np.cross(w_20, X_pos)
    scale_20 = dS_20 * X_pos
    V_2020 = V_94 + dT_20 + cross_20 + scale_20

    return (float(V_2020[0]), float(V_2020[1]), float(V_2020[2]))


def toxyz(lat, lon, h):
    """
    Convert geodetic (lat, lon, h) to ECEF XYZ
    Matches HTDP's TOXYZ subroutine (GRS80).
    """
    a = 6378137.0
    f = 1.0 / 298.257222101
    e2 = 2 * f - f * f

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    N = a / math.sqrt(1 - e2 * sin_lat ** 2)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - e2) + h) * sin_lat

    return x, y, z


def xyz_to_geodetic(x, y, z):
    """
    Convert ECEF XYZ to geodetic (lat, lon, h) on GRS80.
    """
    a = 6378137.0
    f = 1.0 / 298.257222101
    e2 = 2 * f - f * f

    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - e2))

    for _ in range(5):
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * N * math.sin(lat), p)

    lon = math.atan2(y, x)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), h


import numpy as np


def neu_to_xyz_velocity(lat_rad: float, lon_rad: float, v_n_e_u_mm_yr):
    """Convert NEU velocity (mm/yr) to XYZ velocity (mm/yr) at geodetic lat/lon (radians)."""
    vN, vE, vU = map(float, v_n_e_u_mm_yr)
    s, c = np.sin, np.cos
    slat, clat = s(lat_rad), c(lat_rad)
    slon, clon = s(lon_rad), c(lon_rad)

    # Local-to-ECEF rotation (columns = N,E,U in ECEF basis)
    R = np.array([
        [-slat * clon, -slon, clat * clon],
        [-slat * slon, clon, clat * slon],
        [clat, 0.0, slat],
    ], dtype=np.float64)

    v_xyz = R @ np.array([vN, vE, vU], dtype=np.float64)
    return v_xyz  # mm/yr


def xyz_to_neu_velocity(lat_rad: float, lon_rad: float, v_xyz_mm_yr):
    """Convert XYZ velocity (mm/yr) to NEU (mm/yr) at geodetic lat/lon (radians)."""
    s, c = np.sin, np.cos
    slat, clat = s(lat_rad), c(lat_rad)
    slon, clon = s(lon_rad), c(lon_rad)

    R_T = np.array([  # rows = N,E,U in ECEF basis
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [clat * clon, clat * slon, slat],
    ], dtype=np.float64)

    vN, vE, vU = (R_T @ np.array(v_xyz_mm_yr, dtype=np.float64)).tolist()
    return float(vN), float(vE), float(vU)


def apply_htdp_velocity_xyz(lat_in, lon_in, h_in,
                            v_n_grid, v_e_grid, v_up_grid,
                            time_years, height_correction=0.0):
    """
    Apply HTDP velocity using XYZ displacement with optional height correction.

    Args:
        lat_in, lon_in, h_in: Input geodetic coordinates.
        v_n_grid, v_e_grid, v_up_grid: NEU velocities from the grid.
        time_years: Time difference in years.
        height_correction: Correction to ellipsoidal height (default: 0.0).

    Returns:
        Updated geodetic coordinates and XYZ velocities.
    """
    # Apply height correction
    h_corrected = h_in + height_correction

    # 1) Convert to XYZ
    x_in, y_in, z_in = toxyz(lat_in, lon_in, h_corrected)

    # 2) NEU -> XYZ (still in mm/yr)
    vx, vy, vz = neu_to_xyz_velocity(np.radians(lat_in), np.radians(lon_in), (v_n_grid, v_e_grid, v_up_grid))

    # 3) Apply displacement (mm/yr -> m over time)
    dx = (vx / 1000.0) * time_years
    dy = (vy / 1000.0) * time_years
    dz = (vz / 1000.0) * time_years

    x_out = x_in + dx
    y_out = y_in + dy
    z_out = z_in + dz

    # 4) Convert back to geodetic
    lat_out, lon_out, h_out = xyz_to_geodetic(x_out, y_out, z_out)

    return lat_out, lon_out, h_out, (vx, vy, vz)


def vtranf_velocity_xyz(
        X_m, Y_m, Z_m,
        v_xyz_mm_yr,
        dT_mm_yr,  # (dTx, dTy, dTz)  translation RATE, mm/yr
        w_mas_yr,  # (wRx, wRy, wRz)  rotation RATE, milliarcsec/yr
        ds_ppb_yr  # scale RATE, parts-per-billion per year
):
    """
    Transform XYZ velocity (mm/yr) from frame A to frame B using time-rate Helmert:
      v_B = v_A + dT_rate + (Ω_rate x X) + ds_rate * X
    where Ω_rate is the rotation-rate vector (rad/yr) and X is the POSITION (meters).
    All returned in mm/yr.
    """
    # Unpack
    vX, vY, vZ = map(float, v_xyz_mm_yr)
    dTx, dTy, dTz = map(float, dT_mm_yr)
    wRx_mas, wRy_mas, wRz_mas = map(float, w_mas_yr)
    ds = float(ds_ppb_yr)

    # Convert units
    mas_to_rad = np.deg2rad(1.0 / 3600_000.0)  # 1 mas = 4.848136811e-9 rad
    w = np.array([wRx_mas, wRy_mas, wRz_mas], dtype=np.float64) * mas_to_rad  # rad/yr
    X = np.array([X_m, Y_m, Z_m], dtype=np.float64)  # meters

    # Ω x X  (rad/yr x m) => m/yr; scale ds (ppb/yr) x X (m) => m/yr
    cross_term_m_yr = np.cross(w, X)
    scale_term_m_yr = (ds * 1e-9) * X

    # Convert those m/yr to mm/yr
    cross_term_mm_yr = 1000.0 * cross_term_m_yr
    scale_term_mm_yr = 1000.0 * scale_term_m_yr

    vA = np.array([vX, vY, vZ], dtype=np.float64)
    dT = np.array([dTx, dTy, dTz], dtype=np.float64)

    vB = vA + dT + cross_term_mm_yr + scale_term_mm_yr
    return vB.tolist()  # (vX, vY, vZ) in mm/yr


# =========================
# Validation helper (optional)
# =========================
def validate_against_htdp(htdp_grids, test_points=None):
    """
    Validate our implementation against known HTDP outputs.

    NOTE: HTDP official run includes the Helmert transformation.
          This validator focuses on velocity application correctness.
    """
    if test_points is None:
        test_points = [
            {
                'name': 'Oregon Coast Test Point',
                'lat': 44.73310437,
                'lon': -124.05725692,
                'h': 5.067,
                'epoch_from': 2010.0,
                'epoch_to': 2020.0,
                # Reference NEU from HTDP print (approx)
                'htdp_v_n': 12.06,
                'htdp_v_e': 9.58,
                'htdp_v_up': -1.50
            }
        ]

    logger.info("\n" + "=" * 80)
    logger.info("HTDP VELOCITY VALIDATION")
    logger.info("=" * 80)

    for test in test_points:
        logger.info(f"\n{test['name']}")
        logger.info("-" * 80)

        lat_in = test['lat']
        lon_in = test['lon']
        h_in = test['h']
        time_years = test['epoch_to'] - test['epoch_from']

        x_in, y_in, z_in = toxyz(lat_in, lon_in, h_in)
        logger.info(f"INPUT (Geodetic): {lat_in:.8f}deg, {lon_in:.8f}deg, h={h_in:.3f} m")
        logger.info(f"INPUT (ECEF): X={x_in:.3f} m, Y={y_in:.3f} m, Z={z_in:.3f} m")

        v_n, v_e, v_up = htdp_grids.get_velocity(lat_in, lon_in)
        logger.info(f"GRID VELOCITY (NEU mm/yr): N={v_n:+.2f}, E={v_e:+.2f}, Up={v_up:+.2f}")

        vx, vy, vz = neu_to_xyz_velocity(lat_in, lon_in, v_n, v_e, v_up)
        logger.info(f"XYZ velocity (mm/yr): X={vx:+.2f}, Y={vy:+.2f}, Z={vz:+.2f}")

        lat_out, lon_out, h_out, _ = apply_htdp_velocity_xyz(
            lat_in, lon_in, h_in, v_n, v_e, v_up, time_years
        )
        x_out, y_out, z_out = toxyz(lat_out, lon_out, h_out)

        dx = x_out - x_in
        dy = y_out - y_in
        dz = z_out - z_in
        logger.info(f"XYZ displacement over {time_years:.2f} yr: dX={dx:+.3f} m, dY={dy:+.3f} m, dZ={dz:+.3f} m")

    logger.info("\n" + "=" * 80)


# =========================
# Legacy stubs (kept for compatibility)
# =========================
def transform_velocity_itrf2008_to_itrf2020(lat, lon, v_n, v_e, v_up, h=0.0):
    """Legacy: velocities are applied in XYZ; keep signature for compatibility."""
    return (v_n, v_e, v_up)


def apply_geodetic_displacement(lat, lon, h, v_n, v_e, v_up, time_years):
    """Deprecated: use apply_htdp_velocity_xyz()."""
    logger.warning("apply_geodetic_displacement() is deprecated - use apply_htdp_velocity_xyz()")
    lat_out, lon_out, h_out, _ = apply_htdp_velocity_xyz(
        lat, lon, h, v_n, v_e, v_up, time_years
    )
    return (lat_out, lon_out, h_out)