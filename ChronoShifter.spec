# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['GUI_transformation_tool.py'],
    pathex=[],
    binaries=[('C:/Users/engsu.ONID/.conda/envs/pythonProject2/Library/bin/pdal.exe', '.')],
    datas=[('SPCS2022_Zone_Definitions.csv', '.'), ('SPCS22_Zone_Bounds_beta.csv', '.'), ('VelocityGrids_Table.csv', '.'), ('plateModels.csv', '.'), ('nsrs_proj.db', '.'), ('initvl.f', '.'), ('corsdata_itrf2020_xyzcomp.txt', '.'), ('chronoshift_flowchart_final.png', '.'), ('geoid12b', 'geoid12b'), ('geoid18', 'geoid18'), ('geoid2022', 'geoid2022')],
    hiddenimports=['PIL._tkinter_finder', 'laspy', 'pyproj', 'pandas', 'numpy', 'geoid_utils', 'htdp_velocity', 'transformation_core', 'transformation_utils', 'state_bounds', 'zone_code_mapping'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChronoShifter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
