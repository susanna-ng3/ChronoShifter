"""
Diagnose and fix LAS file CRS issues for Alaska data
"""

import laspy
import numpy as np
from pyproj import CRS
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def diagnose_las_crs(las_file):
    """Comprehensive CRS diagnosis for LAS file"""

    logger.info("=" * 80)
    logger.info("LAS FILE CRS DIAGNOSIS")
    logger.info("=" * 80)
    logger.info(f"File: {las_file}")
    logger.info("")

    # Read file
    las = laspy.read(las_file)

    # 1. Check header CRS
    logger.info("1. HEADER CRS INFORMATION:")
    logger.info("-" * 40)

    try:
        header_crs = las.header.parse_crs()
        if header_crs:
            logger.info(f"  CRS Name: {header_crs.name}")
            logger.info(f"  CRS Type: {header_crs.type_name}")
            logger.info(f"  EPSG Code: {header_crs.to_epsg()}")
            logger.info(f"  Is Geographic: {header_crs.is_geographic}")
            logger.info(f"  Is Projected: {header_crs.is_projected}")
        else:
            logger.info("  ⚠ No CRS defined in header")
    except Exception as e:
        logger.info(f"  ✗ Error parsing CRS: {e}")

    logger.info("")

    # 2. Check coordinate ranges
    logger.info("2. COORDINATE RANGES:")
    logger.info("-" * 40)
    logger.info(f"  X: min={las.x.min():.3f}, max={las.x.max():.3f}, range={las.x.max() - las.x.min():.3f}")
    logger.info(f"  Y: min={las.y.min():.3f}, max={las.y.max():.3f}, range={las.y.max() - las.y.min():.3f}")
    logger.info(f"  Z: min={las.z.min():.3f}, max={las.z.max():.3f}, range={las.z.max() - las.z.min():.3f}")
    logger.info("")

    # 3. Diagnose likely CRS based on coordinate ranges
    logger.info("3. CRS DIAGNOSIS:")
    logger.info("-" * 40)

    x_min, x_max = las.x.min(), las.x.max()
    y_min, y_max = las.y.min(), las.y.max()

    # Check if coordinates look like State Plane
    if 100000 < x_min < 1000000 and 1000000 < y_min < 10000000:
        logger.info("  ✓ Coordinates look like Alaska State Plane (projected)")
        logger.info(f"    Expected EPSG: 26934 (NAD83 Alaska Zone 4)")
        logger.info(f"    or EPSG: 6394 (NAD83(2011) Alaska Zone 4)")

    # Check if coordinates look like geographic
    elif -180 < x_min < 180 and -90 < y_min < 90:
        logger.info("  ✓ Coordinates look like Geographic (lat/lon)")
        logger.info(f"    Longitude range: {x_min:.6f} to {x_max:.6f}")
        logger.info(f"    Latitude range: {y_min:.6f} to {y_max:.6f}")

    # Coordinates are wrong
    else:
        logger.info("  ✗ COORDINATES ARE CORRUPTED OR INVALID")
        logger.info(f"    X range: {x_min} to {x_max}")
        logger.info(f"    Y range: {y_min} to {y_max}")
        logger.info("    These don't match any expected coordinate system")

    logger.info("")

    # 4. Check scale and offset
    logger.info("4. SCALE AND OFFSET:")
    logger.info("-" * 40)
    logger.info(f"  X scale: {las.header.x_scale}, offset: {las.header.x_offset}")
    logger.info(f"  Y scale: {las.header.y_scale}, offset: {las.header.y_offset}")
    logger.info(f"  Z scale: {las.header.z_scale}, offset: {las.header.z_offset}")
    logger.info("")

    # 5. Sample points
    logger.info("5. SAMPLE POINTS (first 5):")
    logger.info("-" * 40)
    for i in range(min(5, len(las.x))):
        logger.info(f"  Point {i}: X={las.x[i]:.3f}, Y={las.y[i]:.3f}, Z={las.z[i]:.3f}")

    logger.info("")
    logger.info("=" * 80)

    return las


def fix_las_crs(input_file, output_file, correct_epsg):
    """
    Fix LAS file CRS by setting correct EPSG code

    Args:
        input_file: Input LAZ file with wrong CRS
        output_file: Output LAZ file with correct CRS
        correct_epsg: Correct EPSG code (e.g., 26934 for NAD83 Alaska Zone 4)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("FIXING LAS CRS")
    logger.info("=" * 80)

    # Read file
    las = laspy.read(input_file)

    logger.info(f"Input file: {input_file}")
    logger.info(f"Points: {len(las.points)}")
    logger.info("")

    # Get correct CRS
    correct_crs = CRS.from_epsg(correct_epsg)
    logger.info(f"Setting CRS to: {correct_crs.name} (EPSG:{correct_epsg})")
    logger.info(f"  Type: {correct_crs.type_name}")
    logger.info(f"  Is Projected: {correct_crs.is_projected}")
    logger.info("")

    # Create new header with correct CRS
    las.header.parse_crs = lambda: correct_crs

    # Write output
    logger.info(f"Writing corrected file: {output_file}")
    las.write(output_file)
    logger.info("✓ Complete")
    logger.info("=" * 80)


def check_if_coordinates_make_sense(las_file, expected_epsg):
    """
    Verify if coordinates match the expected CRS

    Args:
        las_file: LAZ file to check
        expected_epsg: Expected EPSG code
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("COORDINATE VALIDATION")
    logger.info("=" * 80)

    las = laspy.read(las_file)
    expected_crs = CRS.from_epsg(expected_epsg)

    logger.info(f"Expected CRS: {expected_crs.name} (EPSG:{expected_epsg})")
    logger.info("")

    x_min, x_max = las.x.min(), las.x.max()
    y_min, y_max = las.y.min(), las.y.max()

    # Define expected ranges for common Alaska CRS
    expected_ranges = {
        26934: {  # NAD83 Alaska Zone 4
            'x': (400000, 800000),
            'y': (5000000, 8000000),
            'name': 'NAD83 Alaska Zone 4'
        },
        6394: {  # NAD83(2011) Alaska Zone 4
            'x': (400000, 800000),
            'y': (5000000, 8000000),
            'name': 'NAD83(2011) Alaska Zone 4'
        },
        6319: {  # NAD83(2011) Geographic
            'x': (-180, -130),  # Longitude for Alaska
            'y': (50, 72),  # Latitude for Alaska
            'name': 'NAD83(2011) Geographic'
        }
    }

    if expected_epsg in expected_ranges:
        expected = expected_ranges[expected_epsg]
        logger.info(f"Expected coordinate ranges for {expected['name']}:")
        logger.info(f"  X: {expected['x'][0]} to {expected['x'][1]}")
        logger.info(f"  Y: {expected['y'][0]} to {expected['y'][1]}")
        logger.info("")
        logger.info(f"Actual coordinate ranges:")
        logger.info(f"  X: {x_min:.3f} to {x_max:.3f}")
        logger.info(f"  Y: {y_min:.3f} to {y_max:.3f}")
        logger.info("")

        # Check if in range
        x_ok = expected['x'][0] <= x_min and x_max <= expected['x'][1]
        y_ok = expected['y'][0] <= y_min and y_max <= expected['y'][1]

        if x_ok and y_ok:
            logger.info("✓ Coordinates are within expected range")
        else:
            logger.info("✗ Coordinates are OUTSIDE expected range")
            if not x_ok:
                logger.info(f"  ✗ X coordinates out of range")
            if not y_ok:
                logger.info(f"  ✗ Y coordinates out of range")
    else:
        logger.info(f"⚠ No validation criteria for EPSG:{expected_epsg}")

    logger.info("=" * 80)


def main():
    """Run CRS diagnostic on Denali dataset"""

    # Your input file
    INPUT_FILE = r"G:\Research\GlitterGulch\PointReg\unique_CC\PPK\PPK_CSF1_unique\SPCS22_AK_Denali_Hwy\scans\GG241_2015_1506_20150530_LPPK.laz"

    # Step 1: Diagnose the issue
    logger.info("\n")
    las = diagnose_las_crs(INPUT_FILE)

    # Step 2: Ask user what to do
    logger.info("\n")
    logger.info("RECOMMENDED ACTIONS:")
    logger.info("=" * 80)

    # Analyze what we found
    x_min, x_max = las.x.min(), las.x.max()
    y_min, y_max = las.y.min(), las.y.max()

    # Case 1: Coordinates look like State Plane but header says geographic
    if 100000 < abs(x_min) < 1000000 and 1000000 < abs(y_min) < 10000000:
        logger.info("Your coordinates LOOK LIKE State Plane, but header may be wrong.")
        logger.info("")
        logger.info("OPTION 1: Fix the header CRS")
        logger.info("  Run: fix_las_crs(INPUT_FILE, 'fixed.laz', 26934)")
        logger.info("  This sets EPSG:26934 (NAD83 Alaska Zone 4)")
        logger.info("")
        logger.info("OPTION 2: If coordinates are NAD83(2011), use EPSG:6394")
        logger.info("  Run: fix_las_crs(INPUT_FILE, 'fixed.laz', 6394)")

    # Case 2: Coordinates are corrupted/wrong
    elif abs(x_min) < 1000 and abs(y_min) < 1000:
        logger.info("✗ YOUR COORDINATES APPEAR CORRUPTED")
        logger.info("")
        logger.info("Possible causes:")
        logger.info("  1. File was reprojected incorrectly")
        logger.info("  2. Scale/offset values are wrong")
        logger.info("  3. Source data was corrupted")
        logger.info("")
        logger.info("RECOMMENDED: Go back to original raw data file")
        logger.info("  Check the original PPK output file before any processing")

    # Case 3: Coordinates look geographic
    elif -180 < x_min < 180 and -90 < y_min < 90:
        logger.info("Your coordinates look like Geographic (lat/lon).")
        logger.info("")
        logger.info("If they should be NAD83(2011) geographic:")
        logger.info("  Run: fix_las_crs(INPUT_FILE, 'fixed.laz', 6319)")
        logger.info("")
        logger.info("But expected State Plane - check your data source!")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()