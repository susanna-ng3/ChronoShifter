import shutil

pdal_path = shutil.which("pdal")
if pdal_path:
    print(f"PDAL is installed at: {pdal_path}")
else:
    print("PDAL is not found in the system PATH.")