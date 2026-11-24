#!/usr/bin/env python3
"""
CORRECTED: Validate HTDP implementation against 48 US state capitals

CORRECT WORKFLOW:
1. Start with NAD83(2011) @ epoch 2010
2. Transform to ITRF2020 @ epoch 2010 (using Table 5 Helmert)
3. Get ITRF2008 velocity at the ITRF2020@2010 position
4. DO NOT transform the velocity - use it directly in XYZ
5. Apply velocity displacement over 10 years in XYZ
6. Result is ITRF2020 @ epoch 2020
"""

import math
import numpy as np
import sys
import os
from pathlib import Path

# Import from local directory
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from htdp_velocity import (
    HTDPVelocityGrids,
    toxyz,
    xyz_to_geodetic,
    neu_to_xyz_velocity,
    xyz_to_neu_velocity,
)
from transformation_utils import (
    get_htdp_table5_params,
    nad83_to_itrf2020_from_table5_epoch
)


def load_test_data():
    """Load HTDP input/output test data - handles both Windows and Linux paths"""

    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent

    # Input file path
    input_file = script_dir / 'CONUS_capitals_HTDP_input_2010.txt'
    output_file = script_dir / 'CONUS_capitals_HTDP_output_2020.txt'

    print(f"Loading from: {script_dir}")
    print(f"  Input file:  {input_file}")
    print(f"  Output file: {output_file}")

    # Verify files exist
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Expected in same directory as script: {script_dir}"
        )
    if not output_file.exists():
        raise FileNotFoundError(
            f"Output file not found: {output_file}\n"
            f"Expected in same directory as script: {script_dir}"
        )

    # Parse input (NAD83@2010 ECEF)
    input_data = {}
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                z = float(parts[2].strip())
                name = parts[3].strip()
                input_data[name] = (x, y, z)

    # Parse output (ITRF2020@2020 ECEF)
    output_data = {}
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                z = float(parts[2].strip())
                name = parts[3].strip()
                output_data[name] = (x, y, z)

    return input_data, output_data


def validate_full_workflow_corrected(name, X_nad83, Y_nad83, Z_nad83,
                                     X_htdp_out, Y_htdp_out, Z_htdp_out,
                                     grids, t0=2010.0, t1=2020.0):
    """
    CORRECTED WORKFLOW - Don't transform velocities!

    Steps:
    1. NAD83(2011)@2010 → ITRF2020@2010 (Helmert)
    2. Get ITRF2008 velocity at ITRF2020@2010 position
    3. Convert NEU velocity to XYZ velocity
    4. Apply XYZ displacement over time (NO velocity transformation!)
    5. Result is ITRF2020@2020
    """
    dt = t1 - t0

    # Step 1: NAD83(2011)@2010 → ITRF2020@2010 using Table 5
    p2010 = get_htdp_table5_params(epoch=2010.0)
    X_itrf2020_t0, Y_itrf2020_t0, Z_itrf2020_t0 = nad83_to_itrf2020_from_table5_epoch(
        X_nad83, Y_nad83, Z_nad83, p2010
    )

    lat_itrf2020_t0, lon_itrf2020_t0, h_itrf2020_t0 = xyz_to_geodetic(
        X_itrf2020_t0, Y_itrf2020_t0, Z_itrf2020_t0
    )

    # Step 2: Get ITRF2008 velocity at ITRF2020@2010 position
    # The grids are in ITRF2008, but we query them at our ITRF2020 position
    # This is OK because the coordinate difference is small (~cm level)
    vN_itrf2008, vE_itrf2008, vU_itrf2008 = grids.get_velocity(
        lat_itrf2020_t0, lon_itrf2020_t0
    )

    # Step 3: Convert NEU to XYZ velocity (using ITRF2020 position)
    vX, vY, vZ = neu_to_xyz_velocity(
        math.radians(lat_itrf2020_t0),
        math.radians(lon_itrf2020_t0),
        (vN_itrf2008, vE_itrf2008, vU_itrf2008)
    )

    # Step 4: Apply XYZ displacement (NO VELOCITY TRANSFORMATION!)
    # KEY: Use ITRF2008 velocities directly - they represent actual crustal motion
    # which is independent of reference frame
    dX = (vX / 1000.0) * dt  # mm/yr to meters over dt years
    dY = (vY / 1000.0) * dt
    dZ = (vZ / 1000.0) * dt

    X_our = X_itrf2020_t0 + dX
    Y_our = Y_itrf2020_t0 + dY
    Z_our = Z_itrf2020_t0 + dZ

    # Calculate errors
    err_X = (X_our - X_htdp_out) * 1000  # meters to mm
    err_Y = (Y_our - Y_htdp_out) * 1000
    err_Z = (Z_our - Z_htdp_out) * 1000
    err_3D = math.sqrt(err_X ** 2 + err_Y ** 2 + err_Z ** 2)

    # Back-calculate HTDP's implied velocity for comparison
    dX_htdp = X_htdp_out - X_itrf2020_t0
    dY_htdp = Y_htdp_out - Y_itrf2020_t0
    dZ_htdp = Z_htdp_out - Z_itrf2020_t0

    vX_htdp = (dX_htdp / dt) * 1000  # m to mm/yr
    vY_htdp = (dY_htdp / dt) * 1000
    vZ_htdp = (dZ_htdp / dt) * 1000

    # Convert to NEU for comparison
    vN_htdp, vE_htdp, vU_htdp = xyz_to_neu_velocity(
        math.radians(lat_itrf2020_t0),
        math.radians(lon_itrf2020_t0),
        (vX_htdp, vY_htdp, vZ_htdp)
    )

    vN_our, vE_our, vU_our = xyz_to_neu_velocity(
        math.radians(lat_itrf2020_t0),
        math.radians(lon_itrf2020_t0),
        (vX, vY, vZ)
    )

    vel_err_N = vN_our - vN_htdp
    vel_err_E = vE_our - vE_htdp
    vel_err_horiz = math.sqrt(vel_err_N ** 2 + vel_err_E ** 2)

    return {
        'name': name,
        'lat': lat_itrf2020_t0,
        'lon': lon_itrf2020_t0,
        'h': h_itrf2020_t0,
        'vN_grid': vN_our,
        'vE_grid': vE_our,
        'vU_grid': vU_our,
        'vN_htdp': vN_htdp,
        'vE_htdp': vE_htdp,
        'vU_htdp': vU_htdp,
        'vel_err_N': vel_err_N,
        'vel_err_E': vel_err_E,
        'vel_err_horiz': vel_err_horiz,
        'err_X_mm': err_X,
        'err_Y_mm': err_Y,
        'err_Z_mm': err_Z,
        'err_3D_mm': err_3D,
        'X_our': X_our,
        'Y_our': Y_our,
        'Z_our': Z_our,
        'X_htdp': X_htdp_out,
        'Y_htdp': Y_htdp_out,
        'Z_htdp': Z_htdp_out,
    }


def write_detailed_report(results, filename='htdp_validation_report_corrected.txt'):
    """Write detailed validation report to file"""

    errors_3d = [r['err_3D_mm'] for r in results]
    vel_errors_horiz = [r['vel_err_horiz'] for r in results]

    excellent = sum(1 for e in errors_3d if e < 5)
    good = sum(1 for e in errors_3d if 5 <= e < 20)
    acceptable = sum(1 for e in errors_3d if 20 <= e < 50)
    poor = sum(1 for e in errors_3d if e >= 50)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("HTDP VELOCITY IMPLEMENTATION VALIDATION REPORT (CORRECTED)\n")
        f.write("=" * 100 + "\n")
        f.write(f"Method: NO velocity transformation (correct approach)\n")
        f.write(f"Total Points: {len(results)}\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Median 3D Error:     {np.median(errors_3d):.2f} mm  ({np.median(errors_3d) / 10:.2f} mm/yr)\n")
        f.write(f"Mean 3D Error:       {np.mean(errors_3d):.2f} mm  ({np.mean(errors_3d) / 10:.2f} mm/yr)\n")
        f.write(f"Std Dev:             {np.std(errors_3d):.2f} mm\n")
        f.write(f"Min Error:           {np.min(errors_3d):.2f} mm\n")
        f.write(f"Max Error:           {np.max(errors_3d):.2f} mm\n\n")

        f.write("ERROR DISTRIBUTION\n")
        f.write("-" * 100 + "\n")
        f.write(f"✅ Excellent (<5mm):       {excellent} points ({excellent / len(results) * 100:.1f}%)\n")
        f.write(f"✓  Good (5-20mm):          {good} points ({good / len(results) * 100:.1f}%)\n")
        f.write(f"⚠️  Acceptable (20-50mm):  {acceptable} points ({acceptable / len(results) * 100:.1f}%)\n")
        f.write(f"❌ Poor (>50mm):           {poor} points ({poor / len(results) * 100:.1f}%)\n\n")

        f.write("VELOCITY STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Mean Horizontal Velocity Error: {np.mean(vel_errors_horiz):.2f} mm/yr\n")
        f.write(f"Median Horizontal Velocity Error: {np.median(vel_errors_horiz):.2f} mm/yr\n")
        f.write(f"Max Horizontal Velocity Error:  {np.max(vel_errors_horiz):.2f} mm/yr\n\n")

        # Detailed results
        f.write("=" * 100 + "\n")
        f.write("DETAILED CITY-BY-CITY RESULTS\n")
        f.write("=" * 100 + "\n\n")

        sorted_results = sorted(results, key=lambda r: r['err_3D_mm'])

        for i, r in enumerate(sorted_results, 1):
            err = r['err_3D_mm']
            if err < 5:
                quality = "✅ EXCELLENT"
            elif err < 20:
                quality = "✓  GOOD    "
            elif err < 50:
                quality = "⚠️  ACCEPTABLE"
            else:
                quality = "❌ POOR    "

            f.write(f"{i:2d}. {r['name']:<25s} {quality}\n")
            f.write("-" * 100 + "\n")
            f.write(f"    Location:     {r['lat']:8.4f}°N, {r['lon']:9.4f}°E, {r['h']:.1f}m\n")
            f.write(f"    Position Errors (10 years):\n")
            f.write(
                f"      X: {r['err_X_mm']:+7.2f} mm    Y: {r['err_Y_mm']:+7.2f} mm    Z: {r['err_Z_mm']:+7.2f} mm\n")
            f.write(f"      3D Error: {r['err_3D_mm']:6.2f} mm  ({r['err_3D_mm'] / 10:.2f} mm/yr)\n")
            f.write(f"    Velocities (mm/yr):\n")
            f.write(f"                    North        East         Up\n")
            f.write(f"      Our Grid:   {r['vN_grid']:+7.2f}    {r['vE_grid']:+7.2f}    {r['vU_grid']:+7.2f}\n")
            f.write(f"      HTDP:       {r['vN_htdp']:+7.2f}    {r['vE_htdp']:+7.2f}    {r['vU_htdp']:+7.2f}\n")
            f.write(
                f"      Difference: {r['vel_err_N']:+7.2f}    {r['vel_err_E']:+7.2f}    {r['vU_grid'] - r['vU_htdp']:+7.2f}\n")
            f.write(f"      Horiz Error: {r['vel_err_horiz']:6.2f} mm/yr\n\n")

    print(f"\n✅ Detailed report written to: {filename}\n")


def main():
    """Run validation"""
    print("\n" + "=" * 80)
    print("CORRECTED VALIDATION - NO VELOCITY TRANSFORMATION")
    print("=" * 80 + "\n")

    # Load data
    print("Loading test data...")
    try:
        input_data, output_data = load_test_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure these files are in the same directory as this script:")
        print("  - CONUS_capitals_HTDP_input_2010.txt")
        print("  - CONUS_capitals_HTDP_output_2020.txt")
        return

    print(f"Loaded {len(input_data)} input points")
    print(f"Loaded {len(output_data)} output points\n")

    # Load grids
    print("Loading HTDP velocity grids...")
    script_dir = Path(__file__).parent
    initvl_path = script_dir / "initvl.f"

    if not initvl_path.exists():
        print(f"\n❌ ERROR: Velocity grid file not found: {initvl_path}")
        print("\nPlease ensure 'initvl.f' is in the same directory as this script")
        return

    grids = HTDPVelocityGrids(str(initvl_path), overlap_mode="highest", verbose=False)
    grids = HTDPVelocityGrids("initvl.f", vertical_override_mm_per_yr=-0.3)
    print(f"Loaded {len(grids.grids)} velocity grids\n")

    # Validate all points
    results = []

    print("Processing points...")
    for i, name in enumerate(sorted(input_data.keys()), 1):
        if name not in output_data:
            print(f"⚠️  Skipping {name} - no output data")
            continue

        X_in, Y_in, Z_in = input_data[name]
        X_out, Y_out, Z_out = output_data[name]

        result = validate_full_workflow_corrected(
            name, X_in, Y_in, Z_in, X_out, Y_out, Z_out, grids
        )
        results.append(result)

        if i % 10 == 0:
            print(f"  Processed {i}/{len(input_data)} points...")

    # Console summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")

    # Group by error magnitude
    excellent = [r for r in results if r['err_3D_mm'] < 5]
    good = [r for r in results if 5 <= r['err_3D_mm'] < 20]
    acceptable = [r for r in results if 20 <= r['err_3D_mm'] < 50]
    poor = [r for r in results if r['err_3D_mm'] >= 50]

    print(f"Results by 3D position error:")
    print(f"  ✅ Excellent (<5mm):     {len(excellent):2d} points ({len(excellent) / len(results) * 100:.1f}%)")
    print(f"  ✓  Good (5-20mm):        {len(good):2d} points ({len(good) / len(results) * 100:.1f}%)")
    print(f"  ⚠️  Acceptable (20-50mm):  {len(acceptable):2d} points ({len(acceptable) / len(results) * 100:.1f}%)")
    print(f"  ❌ Poor (>50mm):         {len(poor):2d} points ({len(poor) / len(results) * 100:.1f}%)\n")

    # Statistics
    errors_3d = [r['err_3D_mm'] for r in results]
    vel_errors = [r['vel_err_horiz'] for r in results]

    median_err = np.median(errors_3d)
    mean_err = np.mean(errors_3d)
    max_err = np.max(errors_3d)

    print(f"Position Error Statistics:")
    print(f"  Median: {median_err:.2f} mm ({median_err / 10:.2f} mm/yr)")
    print(f"  Mean:   {mean_err:.2f} mm ({mean_err / 10:.2f} mm/yr)")
    print(f"  Max:    {max_err:.2f} mm ({max_err / 10:.2f} mm/yr)\n")

    print(f"Velocity Error Statistics:")
    print(f"  Median horizontal: {np.median(vel_errors):.2f} mm/yr")
    print(f"  Mean horizontal:   {np.mean(vel_errors):.2f} mm/yr")
    print(f"  Max horizontal:    {np.max(vel_errors):.2f} mm/yr\n")

    # Show worst 5 cases
    print("=" * 80)
    print("WORST 5 CASES")
    print("=" * 80 + "\n")

    sorted_results = sorted(results, key=lambda r: r['err_3D_mm'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i}. {r['name']}: {r['err_3D_mm']:.2f} mm")
        print(f"   Position: {r['lat']:.4f}°N, {r['lon']:.4f}°E")
        print(f"   Velocity error: N={r['vel_err_N']:+.2f}, E={r['vel_err_E']:+.2f} mm/yr")
        print()

    # Write detailed report
    write_detailed_report(results)

    # Final verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80 + "\n")

    if median_err < 5 and mean_err < 10:
        print("✅ EXCELLENT: Implementation matches HTDP within survey precision!")
    elif median_err < 20 and mean_err < 30:
        print("✓ GOOD: Small discrepancies likely due to grid interpolation")
    elif median_err < 50:
        print("⚠️  ACCEPTABLE: Some systematic differences present")
    else:
        print("❌ NEEDS WORK: Significant errors detected")

    print()


if __name__ == "__main__":
    main()