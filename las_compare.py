# check_input_crs.py
import laspy

files = [
    r"G:\Research\SpencerCreek\04_Analysis\PointReg\20210624_CRS\georeferenced_scans\PPK\ORSPN\20210624_SC_50_50_L_reg.laz",
    r"G:\Research\SpencerCreek\04_Analysis\PointReg\20210624_CRS\georeferenced_scans\PPK\ORST\20210624_SC_50_50_L_reg.laz"
]

for f in files:
    print(f"\n{f}")
    las = laspy.read(f)
    print(f"  Points: {len(las.points):,}")
    print(f"  X range: {las.x.min():.2f} to {las.x.max():.2f}")
    print(f"  Y range: {las.y.min():.2f} to {las.y.max():.2f}")

    # Check for CRS info in VLRs
    if hasattr(las, 'header') and hasattr(las.header, 'vlrs'):
        for vlr in las.header.vlrs:
            if vlr.record_id in [2111, 2112, 34735]:
                print(f"  Found CRS VLR: {vlr.record_id}")