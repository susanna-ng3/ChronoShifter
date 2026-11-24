#!/usr/bin/env python3
"""
Enhanced Geoid utilities for SPCS2022 transformation
Supports both GeoTIFF (preferred with PROJ/PDAL) and Binary formats

Author: Susanna Eng, Michael Olsen
Date: October 2025 - Enhanced for NATRF2022
"""

import os
import struct
import math
import json
import tempfile
import subprocess
import shutil
import logging
import laspy
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path


# ==============================================================================
# OPTIMIZED BINARY GEOID READER (NumPy)
# ==============================================================================

@dataclass
class GeoidBinHeader:
    slat: float
    wlon: float
    dlat: float
    dlon: float
    nlat: int
    nlon: int
    ikind: int


class NGSGeoidBin:
    """NGS Binary Geoid File Reader (Optimized with NumPy)"""

    def __init__(self, path: str):
        self.path = path
        self.header: Optional[GeoidBinHeader] = None
        self._grid: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        """Load geoid file header and grid using NumPy"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Geoid file not found: {self.path}")

        try:
            with open(self.path, 'rb') as f:
                # 1. Detect Endianness
                header_bytes = f.read(44)

                # Try Big Endian (Standard NGS)
                slat, wlon, dlat, dlon, nlat, nlon, ikind = struct.unpack('>4d3i', header_bytes)
                byte_order = '>'

                # Sanity check values
                if not (-90 <= slat <= 90 and 0 <= dlat <= 1 and 0 < nlat < 100000):
                    # Try Little Endian
                    slat, wlon, dlat, dlon, nlat, nlon, ikind = struct.unpack('<4d3i', header_bytes)
                    byte_order = '<'

                self.header = GeoidBinHeader(slat, wlon, dlat, dlon, nlat, nlon, ikind)

                # 2. Read Grid Data with NumPy
                f.seek(44)
                count = nlat * nlon

                dt = np.dtype(f'{byte_order}f4')  # 32-bit float
                self._grid = np.fromfile(f, dtype=dt, count=count)

                if self._grid.size != count:
                    raise ValueError(f"Incomplete grid: expected {count}, got {self._grid.size}")

                # Reshape into (nlat, nlon)
                self._grid = self._grid.reshape((nlat, nlon))

        except Exception as e:
            raise ValueError(f"Failed to load geoid file {self.path}: {str(e)}")

    def sample(self, lat_deg: float, lon_deg: float) -> float:
        """Sample geoid height with Bilinear Interpolation"""
        h = self.header

        lon = lon_deg

        # Handle 0–360 wrapping
        grid_max_lon = h.wlon + (h.nlon - 1) * h.dlon

        if grid_max_lon > 180 and lon < 0:
            lon += 360.0
        elif h.wlon < 0 and lon > 180:
            lon -= 360.0

        # Fractional rows/columns
        row = (lat_deg - h.slat) / h.dlat
        col = (lon - h.wlon) / h.dlon

        if row < 0 or row > h.nlat - 1.00001:
            return float('nan')
        if col < 0 or col > h.nlon - 1.00001:
            return float('nan')

        r0 = int(row)
        c0 = int(col)

        # Clamp
        r0 = min(r0, h.nlat - 2)
        c0 = min(c0, h.nlon - 2)

        r1, c1 = r0 + 1, c0 + 1

        dr = row - r0
        dc = col - c0

        v00 = self._grid[r0, c0]
        v01 = self._grid[r0, c1]
        v10 = self._grid[r1, c0]
        v11 = self._grid[r1, c1]

        # Bilinear interpolation
        top = v00 * (1 - dc) + v01 * dc
        bot = v10 * (1 - dc) + v11 * dc
        val = top * (1 - dr) + bot * dr

        return float(val)


class NGSGeoidBin:
    """NGS Binary Geoid File Reader"""

    def __init__(self, path: str):
        self.path = path
        self.header: Optional[GeoidBinHeader] = None
        self._grid: Optional[List[float]] = None
        self.byte_order = '<'
        self._debug_samples = 0
        self._load()

    def _detect_endianness(self):
        """Detect byte order"""
        with open(self.path, 'rb') as f:
            test_bytes = f.read(44)

            if len(test_bytes) < 44:
                logging.warning(f"File too short: {self.path}")
                return False

            for byte_order in ['<', '>']:
                try:
                    header_fmt = f'{byte_order}4d3i'
                    slat, wlon, dlat, dlon, nlat, nlon, ikind = struct.unpack(header_fmt, test_bytes)

                    if (-90 <= slat <= 90 and -180 <= wlon <= 360 and
                            0 < dlat < 1 and 0 < dlon < 1 and
                            100 < nlat < 50000 and 100 < nlon < 50000 and ikind == 1):
                        self.byte_order = byte_order
                        endian_name = "big-endian" if byte_order == '>' else "little-endian"
                        logging.info(f"Detected {endian_name} format for {os.path.basename(self.path)}")
                        return True

                except struct.error:
                    continue

            self.byte_order = '<'
            return False

    def _load(self):
        """Load geoid file"""
        self._detect_endianness()

        try:
            with open(self.path, 'rb') as f:
                header_fmt = f'{self.byte_order}4d3i'
                header_bytes = f.read(struct.calcsize(header_fmt))

                if len(header_bytes) != struct.calcsize(header_fmt):
                    raise ValueError('Incomplete header')

                slat, wlon, dlat, dlon, nlat, nlon, ikind = struct.unpack(header_fmt, header_bytes)

                if not (-90 <= slat <= 90 and -180 <= wlon <= 360 and
                        0 < dlat < 1 and 0 < dlon < 1 and
                        100 < nlat < 50000 and 100 < nlon < 50000 and ikind == 1):
                    raise ValueError('Invalid header values')

                self.header = GeoidBinHeader(slat, wlon, dlat, dlon, nlat, nlon, ikind)

                count = nlat * nlon
                data_fmt = f'{self.byte_order}{count}f'
                data_bytes = f.read(struct.calcsize(data_fmt))

                if len(data_bytes) != struct.calcsize(data_fmt):
                    raise ValueError('Incomplete grid data')

                self._grid = list(struct.unpack(data_fmt, data_bytes))

                logging.info(f"Loaded geoid grid: {nlat}×{nlon} points")

        except Exception as e:
            raise ValueError(f'Failed to load geoid file {self.path}: {str(e)}')

    def sample(self, lat_deg: float, lon_deg: float) -> float:
        """Sample geoid height with bilinear interpolation"""
        try:
            h = self.header

            # Convert longitude to match file convention
            lon = lon_deg

            # Calculate eastern boundary
            east_lon = h.wlon + (h.nlon - 1) * h.dlon

            # If eastern boundary > 180, file uses 0-360 convention
            if east_lon > 180:
                if lon < 0:
                    lon += 360  # Convert -180/+180 to 0-360
            elif h.wlon < 0:  # File uses -180/+180 convention
                if lon > 180:
                    lon -= 360  # Convert 0-360 to -180/+180

            # Calculate fractional indices
            fi = (lat_deg - h.slat) / h.dlat
            fj = (lon - h.wlon) / h.dlon

            # Check bounds with small tolerance for floating point errors
            if fi < 0 or fj < 0 or fi >= h.nlat - 1 or fj >= h.nlon - 1:
                return float('nan')

            # Get integer indices
            i0, j0 = int(fi), int(fj)
            i1, j1 = min(i0 + 1, h.nlat - 1), min(j0 + 1, h.nlon - 1)

            # Interpolation weights
            di, dj = fi - i0, fj - j0

            # Get grid values
            def get_val(i, j):
                return self._grid[i * h.nlon + j]

            # Bilinear interpolation
            q11 = get_val(i0, j0)
            q21 = get_val(i1, j0)
            q12 = get_val(i0, j1)
            q22 = get_val(i1, j1)

            r1 = q11 * (1 - dj) + q12 * dj
            r2 = q21 * (1 - dj) + q22 * dj
            result = r1 * (1 - di) + r2 * di

            return float(result)

        except Exception:
            return float('nan')


# ==============================================================================
# ENHANCED GEOID CORRECTOR (GeoTIFF + PROJ/PDAL with Binary Fallback)
# ==============================================================================

class EnhancedGeoidCorrector:
    """
    Unified geoid correction handler with GeoTIFF+PROJ primary method,
    binary fallback for reliability.

    Designed for NATRF2022 transformations with SPCS2022.
    """

    def __init__(self,
                 geoid22_tif=None, geoid18_tif=None,
                 geoid22_bin=None, geoid18_bin=None, geoid12b_bin=None):
        """
        Initialize with paths to GeoTIFF (preferred) and binary (fallback) geoid files.

        Args:
            geoid22_tif: Path to GEOID22 GeoTIFF (e.g., us_noaa_sgeoid2022_na_v1a.tif)
            geoid18_tif: Path to GEOID18 GeoTIFF (e.g., us_noaa_g2018u0.tif)
            geoid22_bin: Path to GEOID22 binary (e.g., SGEOID2022.NA.N.beta_v0.bin)
            geoid18_bin: Path to GEOID18 binary (e.g., g2018u1.bin)
            geoid12b_bin: Path to GEOID12B binary (legacy support)
        """
        self.geoid22_tif = geoid22_tif
        self.geoid18_tif = geoid18_tif
        self.geoid22_bin = geoid22_bin
        self.geoid18_bin = geoid18_bin
        self.geoid12b_bin = geoid12b_bin

        self.method_used = None
        self.pdal_available = self._check_pdal_available()

        # Lazy-load binary readers only if needed
        self._geoid22_reader = None
        self._geoid18_reader = None
        self._geoid12b_reader = None

    def _check_pdal_available(self) -> bool:
        """Check if PDAL is available for GeoTIFF processing"""
        pdal_exe = shutil.which('pdal')
        if pdal_exe:
            logging.info(f"✓ PDAL found at: {pdal_exe}")
            return True
        else:
            logging.warning("⚠ PDAL not found - will use binary method")
            return False

    def apply_geoid_correction(self, input_file, output_file,
                               use_geoid22=False,
                               remove_geoid18=False,
                               remove_geoid12b=False,
                               progress_callback=None):
        """
        Apply geoid corrections using best available method.

        Strategy:
        1. Try GeoTIFF + PROJ/PDAL if files available and PDAL installed
        2. Fall back to binary method if GeoTIFF fails or unavailable

        Args:
            input_file: Input LAS/LAZ file path
            output_file: Output LAS/LAZ file path
            use_geoid22: Apply GEOID22 (for NAD83 → NATRF2022/NAPGD2022)
            remove_geoid18: Remove GEOID18 (for NAVD88 → ellipsoid)
            remove_geoid12b: Remove GEOID12B (legacy, if needed)
            progress_callback: Optional callback(progress, message)

        Returns:
            bool: True if successful
        """
        logging.info("=" * 70)
        logging.info("GEOID CORRECTION FOR NATRF2022 TRANSFORMATION")
        logging.info("=" * 70)

        if progress_callback:
            progress_callback(5, "Initializing geoid correction...")

        # Determine which method to use
        use_geotiff = (
                self.pdal_available and
                ((use_geoid22 and self.geoid22_tif and os.path.exists(self.geoid22_tif)) or
                 (remove_geoid18 and self.geoid18_tif and os.path.exists(self.geoid18_tif)))
        )

        try:
            if use_geotiff:
                logging.info("Using GeoTIFF + PROJ/PDAL method...")
                self._apply_with_proj(
                    input_file, output_file,
                    use_geoid22, remove_geoid18, remove_geoid12b,
                    progress_callback
                )
                self.method_used = "GeoTIFF+PROJ"
                logging.info("✓ GeoTIFF + PROJ method succeeded")
                return True
            else:
                # Go straight to binary if GeoTIFF not available
                raise ValueError("GeoTIFF not available, using binary method")

        except Exception as e:
            logging.warning(f"GeoTIFF + PROJ failed or unavailable: {e}")
            logging.info("Falling back to binary method...")

            try:
                self._apply_with_binary(
                    input_file, output_file,
                    use_geoid22, remove_geoid18, remove_geoid12b,
                    progress_callback
                )
                self.method_used = "Binary"
                logging.info("✓ Binary method succeeded")
                return True

            except Exception as e2:
                logging.error(f"Both methods failed!")
                logging.error(f"  GeoTIFF error: {e}")
                logging.error(f"  Binary error: {e2}")
                self.method_used = "FAILED"
                if progress_callback:
                    progress_callback(0, "Geoid correction failed!")
                return False

    def _apply_with_proj(self, input_file, output_file,
                         use_geoid22, remove_geoid18, remove_geoid12b,
                         progress_callback=None):
        """Apply geoid correction using PROJ/PDAL with GeoTIFF grids"""

        if progress_callback:
            progress_callback(10, "Building PDAL pipeline for GeoTIFF...")

        # Build PDAL pipeline
        operations = []
        temp_files = []
        current_input = input_file

        try:
            # Step 1: Remove GEOID12B if requested (must use binary)
            if remove_geoid12b and self.geoid12b_bin:
                logging.info("  Removing GEOID12B (binary method)...")
                temp_12b = tempfile.NamedTemporaryFile(suffix='.laz', delete=False).name
                temp_files.append(temp_12b)

                self._load_binary_reader('geoid12b')
                apply_geoid_corrections_laspy(
                    current_input, temp_12b,
                    None, None, self._geoid12b_reader,
                    False, False, True,
                    progress_callback
                )
                current_input = temp_12b

            # Step 2: Remove GEOID18 using GeoTIFF
            if remove_geoid18:
                if not self.geoid18_tif or not os.path.exists(self.geoid18_tif):
                    raise FileNotFoundError(f"GEOID18 GeoTIFF not found: {self.geoid18_tif}")

                logging.info(f"  Removing GEOID18: {os.path.basename(self.geoid18_tif)}")
                temp_18 = tempfile.NamedTemporaryFile(suffix='.laz', delete=False).name
                temp_files.append(temp_18)

                # GEOID18 removal: NAVD88 (orthometric) → NAD83(2011) (ellipsoid)
                # This adds the geoid height back (orthometric + geoid = ellipsoid)
                pipeline = {
                    "pipeline": [
                        current_input,
                        {
                            "type": "filters.reprojection",
                            "in_srs": "EPSG:6318+5703",  # NAD83(2011) + NAVD88
                            "out_srs": "EPSG:6318"  # NAD83(2011) ellipsoid
                        },
                        {
                            "type": "writers.las",
                            "filename": temp_18,
                            "forward": "all"
                        }
                    ]
                }

                if progress_callback:
                    progress_callback(40, "Removing GEOID18 with PROJ...")
                self._run_pdal_pipeline(pipeline)
                current_input = temp_18

            # Step 3: Apply GEOID22 using GeoTIFF
            if use_geoid22:
                if not self.geoid22_tif or not os.path.exists(self.geoid22_tif):
                    raise FileNotFoundError(f"GEOID22 GeoTIFF not found: {self.geoid22_tif}")

                logging.info(f"  Applying GEOID22: {os.path.basename(self.geoid22_tif)}")

                # GEOID22 application: NAD83(2011) (ellipsoid) → NAPGD2022 (orthometric)
                # Use custom PROJ string with geoidgrids parameter
                pipeline = {
                    "pipeline": [
                        current_input,
                        {
                            "type": "filters.reprojection",
                            "in_srs": "EPSG:6318",  # NAD83(2011) ellipsoid
                            "out_srs": f"+proj=longlat +datum=NAD83 +geoidgrids={self.geoid22_tif} +vunits=m +no_defs"
                        },
                        {
                            "type": "writers.las",
                            "filename": output_file,
                            "forward": "all"
                        }
                    ]
                }

                if progress_callback:
                    progress_callback(70, "Applying GEOID22 with PROJ...")
                self._run_pdal_pipeline(pipeline)
            else:
                # No GEOID22, just copy the result
                shutil.copy2(current_input, output_file)

            if progress_callback:
                progress_callback(100, "GeoTIFF geoid correction completed!")

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

    def _apply_with_binary(self, input_file, output_file,
                           use_geoid22, remove_geoid18, remove_geoid12b,
                           progress_callback=None):
        """Fallback: Apply geoid correction using binary files"""

        logging.info("Using binary geoid correction method")

        if progress_callback:
            progress_callback(10, "Loading binary geoid files...")

        # Load binary readers as needed
        geoid22_reader = None
        geoid18_reader = None
        geoid12b_reader = None

        if use_geoid22:
            if not self.geoid22_bin or not os.path.exists(self.geoid22_bin):
                raise FileNotFoundError(f"GEOID22 binary not found: {self.geoid22_bin}")
            self._load_binary_reader('geoid22')
            geoid22_reader = self._geoid22_reader

        if remove_geoid18:
            if not self.geoid18_bin or not os.path.exists(self.geoid18_bin):
                raise FileNotFoundError(f"GEOID18 binary not found: {self.geoid18_bin}")
            self._load_binary_reader('geoid18')
            geoid18_reader = self._geoid18_reader

        if remove_geoid12b:
            if not self.geoid12b_bin or not os.path.exists(self.geoid12b_bin):
                raise FileNotFoundError(f"GEOID12B binary not found: {self.geoid12b_bin}")
            self._load_binary_reader('geoid12b')
            geoid12b_reader = self._geoid12b_reader

        # Use existing binary method
        apply_geoid_corrections_laspy(
            input_file, output_file,
            geoid22_reader, geoid18_reader, geoid12b_reader,
            use_geoid22, remove_geoid18, remove_geoid12b,
            progress_callback
        )

    def _load_binary_reader(self, geoid_type: str):
        """Lazy-load binary geoid readers and return the reader object"""
        if geoid_type == 'geoid22':
            if not self._geoid22_reader:
                logging.info(f"Loading GEOID22 binary: {os.path.basename(self.geoid22_bin)}")
                self._geoid22_reader = NGSGeoidBin(self.geoid22_bin)
            return self._geoid22_reader
        elif geoid_type == 'geoid18':
            if not self._geoid18_reader:
                logging.info(f"Loading GEOID18 binary: {os.path.basename(self.geoid18_bin)}")
                self._geoid18_reader = NGSGeoidBin(self.geoid18_bin)
            return self._geoid18_reader
        elif geoid_type == 'geoid12b':
            if not self._geoid12b_reader:
                logging.info(f"Loading GEOID12B binary: {os.path.basename(self.geoid12b_bin)}")
                self._geoid12b_reader = NGSGeoidBin(self.geoid12b_bin)
            return self._geoid12b_reader
        return None

    def _run_pdal_pipeline(self, pipeline_dict):
        """Execute a PDAL pipeline"""
        # Write pipeline to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump(pipeline_dict, f, indent=2)
            pipeline_file = f.name

        try:
            # Get PDAL executable
            pdal_exe = shutil.which('pdal')
            if not pdal_exe:
                raise FileNotFoundError("PDAL executable not found")

            # Run PDAL
            cmd = [pdal_exe, 'pipeline', pipeline_file]
            logging.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Log PDAL output at debug level
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logging.debug(f"PDAL: {line}")

        except subprocess.CalledProcessError as e:
            logging.error(f"PDAL pipeline failed: {e.stderr}")
            raise
        finally:
            # Clean up temp pipeline file
            if os.path.exists(pipeline_file):
                try:
                    os.unlink(pipeline_file)
                except:
                    pass


def _sample_geoid_array(reader, lat_arr, lon_arr):
    """
    Vectorized bilinear sampling of an NGSGeoidBin over arrays of lat/lon.

    Parameters
    ----------
    reader : NGSGeoidBin
        Geoid reader instance with .header and ._grid
    lat_arr : np.ndarray
        Latitudes in degrees (same convention as scalar sample)
    lon_arr : np.ndarray
        Longitudes in degrees

    Returns
    -------
    np.ndarray
        Geoid heights (same shape as lat_arr), NaN where out of bounds.
    """
    h = reader.header
    if h is None:
        raise ValueError("Geoid reader header is not initialized")

    # Ensure numpy arrays
    lat = np.asarray(lat_arr, dtype=float)
    lon = np.asarray(lon_arr, dtype=float)

    # Reshape grid to (nlat, nlon) regardless of original storage
    grid = np.asarray(reader._grid, dtype=float).reshape(h.nlat, h.nlon)

    # Handle longitude conventions (0–360 vs -180–180), matching scalar logic
    lon_adj = lon.copy()
    east_lon = h.wlon + (h.nlon - 1) * h.dlon

    if east_lon > 180:
        # File uses 0–360; convert negative longitudes
        lon_adj = np.where(lon_adj < 0.0, lon_adj + 360.0, lon_adj)
    elif h.wlon < 0:
        # File uses -180–180; convert > 180 down
        lon_adj = np.where(lon_adj > 180.0, lon_adj - 360.0, lon_adj)

    # Fractional indices
    fi = (lat - h.slat) / h.dlat
    fj = (lon_adj - h.wlon) / h.dlon

    # Valid mask (mimic scalar bounds)
    valid = (
        (fi >= 0.0) & (fj >= 0.0) &
        (fi < (h.nlat - 1)) &
        (fj < (h.nlon - 1))
    )

    # Initialize result
    out = np.full(lat.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out

    fi_v = fi[valid]
    fj_v = fj[valid]

    i0 = np.floor(fi_v).astype(int)
    j0 = np.floor(fj_v).astype(int)

    # Clamp to edges so i1, j1 stay valid
    i0 = np.clip(i0, 0, h.nlat - 2)
    j0 = np.clip(j0, 0, h.nlon - 2)

    i1 = i0 + 1
    j1 = j0 + 1

    di = fi_v - i0
    dj = fj_v - j0

    # Bilinear interpolation
    v00 = grid[i0, j0]
    v01 = grid[i0, j1]
    v10 = grid[i1, j0]
    v11 = grid[i1, j1]

    r1 = v00 * (1.0 - dj) + v01 * dj
    r2 = v10 * (1.0 - dj) + v11 * dj
    vals = r1 * (1.0 - di) + r2 * di

    out[valid] = vals
    return out

def apply_geoid_corrections_laspy(input_file, output_file, geoid22_reader, geoid18_reader, geoid12b_reader,
                                  use_geoid22, remove_geoid18, remove_geoid12b, progress_callback=None):
    """Apply geoid corrections using laspy directly with version preservation (vectorized)."""

    logging.info("Starting geoid correction using laspy (vectorized)...")
    logging.info(
        f"Corrections to apply: GEOID22={use_geoid22}, Remove GEOID18={remove_geoid18}, Remove GEOID12B={remove_geoid12b}"
    )

    try:
        if progress_callback:
            progress_callback(10, "Reading LAS file...")

        las = laspy.read(input_file)
        original_header = las.header

        logging.info(f"Read {len(las.x):,} points from {input_file}")
        logging.info(f"Input LAS version: {original_header.version.major}.{original_header.version.minor}")
        logging.info(f"Input point format: {original_header.point_format.id}")

        # Coordinates as numpy arrays
        x_coords = np.asarray(las.x, dtype=float)  # lon
        y_coords = np.asarray(las.y, dtype=float)  # lat
        z_orig   = np.asarray(las.z, dtype=float)

        if progress_callback:
            progress_callback(30, "Applying geoid corrections (vectorized)...")

        z_new = z_orig.copy()

        geoid22_corrections = 0
        geoid18_corrections = 0
        geoid12b_corrections = 0
        sample_corrections = []

        # ------------------------
        # 1) Remove GEOID12B
        # ------------------------
        if remove_geoid12b and geoid12b_reader:
            logging.info("Removing GEOID12B using binary grid (vectorized)...")
            n12b = _sample_geoid_array(geoid12b_reader, y_coords, x_coords)
            mask12 = ~np.isnan(n12b)

            if np.any(mask12):
                # Subtract negative file value = add positive undulation
                z_new[mask12] -= n12b[mask12]
                geoid12b_corrections = int(mask12.sum())

                # Log a few samples
                idx_samples = np.flatnonzero(mask12)[:3]
                for k, idx in enumerate(idx_samples, start=1):
                    logging.info(
                        f"GEOID12B removal {k}: lat={y_coords[idx]:.6f}, lon={x_coords[idx]:.6f}, geoid={n12b[idx]:.3f}m"
                    )

            logging.info(f"  GEOID12B removed from {geoid12b_corrections:,} points")

            if progress_callback:
                progress_callback(40, "Removed GEOID12B (vectorized)")

        # ------------------------
        # 2) Remove GEOID18
        # ------------------------
        if remove_geoid18 and geoid18_reader:
            logging.info("Removing GEOID18 using binary grid (vectorized)...")
            n18 = _sample_geoid_array(geoid18_reader, y_coords, x_coords)
            mask18 = ~np.isnan(n18)

            if np.any(mask18):
                # Subtract negative file value = add positive undulation
                z_new[mask18] += n18[mask18]
                geoid18_corrections = int(mask18.sum())

                idx_samples = np.flatnonzero(mask18)[:3]
                for k, idx in enumerate(idx_samples, start=1):
                    logging.info(
                        f"GEOID18 removal {k}: lat={y_coords[idx]:.6f}, lon={x_coords[idx]:.6f}, geoid={n18[idx]:.3f}m"
                    )

            logging.info(f"  GEOID18 removed from {geoid18_corrections:,} points")

            if progress_callback:
                progress_callback(55, "Removed GEOID18 (vectorized)")

        # ------------------------
        # 3) Apply GEOID22
        # ------------------------
        if use_geoid22 and geoid22_reader:
            logging.info("Applying GEOID22 using binary grid (vectorized)...")
            n22 = _sample_geoid_array(geoid22_reader, y_coords, x_coords)
            mask22 = ~np.isnan(n22)

            if np.any(mask22):
                z_new[mask22] -= n22[mask22]
                geoid22_corrections = int(mask22.sum())

                # Collect up to 5 sample corrections for logging
                idx_samples = np.flatnonzero(mask22)[:5]
                for idx in idx_samples:
                    original_height = float(z_orig[idx])
                    corrected_height = float(z_new[idx])
                    sample_corrections.append({
                        'point': int(idx),
                        'lat': float(y_coords[idx]),
                        'lon': float(x_coords[idx]),
                        'original_height': original_height,
                        'geoid22_value': float(n22[idx]),
                        'corrected_height': corrected_height
                    })

                if sample_corrections:
                    logging.info("Sample GEOID22 corrections:")
                    for sample in sample_corrections[:3]:
                        height_change = sample['corrected_height'] - sample['original_height']
                        logging.info(
                            f"  Point {sample['point']}: "
                            f"{sample['original_height']:.3f}m -> {sample['corrected_height']:.3f}m "
                            f"(change: {height_change:.3f}m, geoid22={sample['geoid22_value']:.3f}m)"
                        )

            logging.info(f"  GEOID22 applied to {geoid22_corrections:,} points")

            if progress_callback:
                progress_callback(70, "Applied GEOID22 (vectorized)")

        # ------------------------
        # 4) Summary & write
        # ------------------------
        # Points where |Δh| > 1 mm
        height_delta = z_new - z_orig
        corrections_applied = int(np.count_nonzero(np.abs(height_delta) > 0.001))

        las.z = z_new

        if progress_callback:
            progress_callback(80, "Writing corrected LAS file...")

        las.write(output_file)

        logging.info(f"Applied geoid corrections to {corrections_applied:,} points")
        if remove_geoid12b:
            logging.info(f"  GEOID12B removed from {geoid12b_corrections:,} points")
        if remove_geoid18:
            logging.info(f"  GEOID18 removed from {geoid18_corrections:,} points")
        if use_geoid22:
            logging.info(f"  GEOID22 applied to {geoid22_corrections:,} points")

        logging.info(f"Corrected file saved to: {output_file}")
        logging.info(
            f"Output preserves LAS {original_header.version.major}.{original_header.version.minor}, "
            f"point format {original_header.point_format.id}"
        )

        if progress_callback:
            progress_callback(100, "Geoid correction completed!")

        return True

    except Exception as e:
        logging.error(f"Error in geoid correction: {str(e)}")
        if progress_callback:
            progress_callback(0, "Geoid correction failed!")
        return False


# ==============================================================================
# GUI INTEGRATION HELPERS
# ==============================================================================

def create_geoid_corrector_from_gui(gui):
    """
    Create EnhancedGeoidCorrector from GUI settings.

    Args:
        gui: GUI instance with geoid file paths

    Returns:
        EnhancedGeoidCorrector instance
    """
    return EnhancedGeoidCorrector(
        geoid22_tif=gui.geoid22_tif_path.get() if hasattr(gui, 'geoid22_tif_path') else None,
        geoid18_tif=gui.geoid18_tif_path.get() if hasattr(gui, 'geoid18_tif_path') else None,
        geoid22_bin=gui.geoid22_file_path.get() if gui.geoid22_file_path.get() else None,
        geoid18_bin=gui.geoid18_file_path.get() if gui.geoid18_file_path.get() else None,
        geoid12b_bin=gui.geoid12b_file_path.get() if gui.geoid12b_file_path.get() else None
    )