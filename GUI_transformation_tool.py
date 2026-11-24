#!/usr/bin/env python3
"""
GUI Transformation Tool for NSRS Modernization
NAD83(2011) → ITRF2020/NATRF2022 → SPCS2022
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import time
import sys
import csv
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
from transformation_core import run_enhanced_transformation
from state_bounds import get_state_from_coords

# === FIX FOR PYINSTALLER GUI MODE ===
# When running as GUI (console=False), stdout/stderr are None
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')


# === END FIX ===

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
        logging.debug(f"Checking PyInstaller path: {candidate}")
        if os.path.exists(candidate):
            logging.info(f"Found resource at: {candidate}")
            return candidate

    # 2. Current working directory
    cwd_candidate = os.path.join(os.getcwd(), filename)
    paths_to_check.append(cwd_candidate)
    logging.debug(f"Checking cwd path: {cwd_candidate}")
    if os.path.exists(cwd_candidate):
        logging.info(f"Found resource at: {cwd_candidate}")
        return cwd_candidate

    # 3. Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_candidate = os.path.join(script_dir, filename)
    paths_to_check.append(script_candidate)
    logging.debug(f"Checking script path: {script_candidate}")
    if os.path.exists(script_candidate):
        logging.info(f"Found resource at: {script_candidate}")
        return script_candidate

    # Not found - log all attempted paths
    logging.error(f"Resource '{filename}' not found. Searched:")
    for path in paths_to_check:
        logging.error(f"  - {path}")

    # Return first preference even if not found (will error at open())
    return paths_to_check[0]


def find_pdal_executable():
    """
    Find PDAL executable in system PATH or conda environment.
    Returns path or None if not found.
    """
    import shutil

    # Try standard PATH search first
    pdal_path = shutil.which('pdal')
    if pdal_path:
        logging.info(f"Found PDAL in PATH: {pdal_path}")
        return pdal_path

    # Try conda environment (if running in conda)
    if 'CONDA_PREFIX' in os.environ:
        conda_scripts = os.path.join(os.environ['CONDA_PREFIX'], 'Scripts', 'pdal.exe')
        if os.path.exists(conda_scripts):
            logging.info(f"Found PDAL in conda: {conda_scripts}")
            return conda_scripts

    # Try common installation locations
    common_paths = [
        'C:\\Program Files\\PDAL\\bin\\pdal.exe',
        'C:\\Program Files (x86)\\PDAL\\bin\\pdal.exe',
        os.path.expanduser('~\\miniconda3\\Scripts\\pdal.exe'),
        os.path.expanduser('~\\anaconda3\\Scripts\\pdal.exe'),
    ]

    for path in common_paths:
        if os.path.exists(path):
            logging.info(f"Found PDAL at: {path}")
            return path

    logging.warning("PDAL executable not found. Please install with: conda install pdal")
    return None


class TextHandler(logging.Handler):
    """Custom logging handler to redirect logs to GUI"""

    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)

        self.text_widget.after(0, append)


class TransformationGUI:
    """GUI for coordinate transformation tool"""

    def __init__(self, root):
        self.root = root
        self.root.title("ChronoShifter - an NSRS Modernization Transformation Tool for Point Clouds")
        self.root.geometry("950x850")

        # Variables for Transform tab
        self.input_file_var = tk.StringVar()
        self.input_files = []  # List for batch processing
        self.input_crs_var = tk.StringVar(value="EPSG:6319")  # Default NAD83(2011) geographic
        self.output_file_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.zone_var = tk.StringVar()
        self.velocity_file_var = tk.StringVar()
        self.velocity_method_var = tk.StringVar(value="helmert_only")
        self.source_epoch_var = tk.StringVar(value="2010.0")
        self.target_epoch_var = tk.StringVar(value="2020.0")

        # Variables for Geoid tab
        self.geoid22_file_var = tk.StringVar()
        self.geoid18_file_var = tk.StringVar()
        self.geoid12b_file_var = tk.StringVar()
        self.remove_geoid12b_var = tk.BooleanVar(value=True)
        self.remove_geoid18_var = tk.BooleanVar(value=False)
        self.apply_geoid22_var = tk.BooleanVar(value=True)

        # Progress variables
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")

        # Collapsible section states
        self.section_states = {}

        # Zone data
        self.zones_data = []
        self.zones_by_state = {}

        # Setup logging
        self.setup_logging()

        # Load zone data
        self.load_zone_definitions()

        # Create UI
        self.create_ui()

        # Auto-detect files
        self.auto_detect_input()
        self.auto_detect_geoid_files()

        self.status_var.set("Ready - Select input file")

        # Load zone bounds for auto-detection
        self.zone_bounds_df = None
        self.load_zone_bounds()

    def get_resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and PyInstaller."""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def load_zone_bounds(self):
        """Load zone bounds from CSV for auto-detection."""
        try:
            import pandas as pd
            zone_bounds_path = self.get_resource_path('SPCS22_Zone_Bounds_beta.csv')

            if os.path.exists(zone_bounds_path):
                self.zone_bounds_df = pd.read_csv(zone_bounds_path)
                print(f"DEBUG: Loaded {len(self.zone_bounds_df)} zone bounds")
                return True
            else:
                print(f"WARNING: Zone bounds file not found: {zone_bounds_path}")
                self.zone_bounds_df = None
                return False
        except Exception as e:
            print(f"ERROR loading zone bounds: {e}")
            self.zone_bounds_df = None
            return False

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)  # Assign logger to the class
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False, buffering=1)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # Initialize file handler placeholder (will be set up during transformation)
        self.log_file_handler = None

    def load_zone_definitions(self):
        """Load SPCS2022 zone definitions from CSV"""
        csv_file = get_resource_path("SPCS2022_Zone_Definitions.csv")

        if not os.path.exists(csv_file):
            logging.warning(f"Zone definitions CSV not found: {csv_file}")
            return

        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                self.zones_data = list(reader)

            # Group zones by state
            for zone in self.zones_data:
                zone_abrv = zone.get('Zone Abrv', '').strip()
                zone_name = zone.get('Zone Name', '').strip()
                zone_code = zone.get('Zone Code', '').strip()

                # Extract state abbreviation
                state_abbr = zone_abrv.split('_')[0] if '_' in zone_abrv else zone_abrv[:2]

                # Skip special zones
                if zone_abrv in ['GOM', 'NAVA', 'KANC']:
                    state_abbr = 'SPECIAL'

                if state_abbr not in self.zones_by_state:
                    self.zones_by_state[state_abbr] = []

                self.zones_by_state[state_abbr].append({
                    'code': zone_code,
                    'abrv': zone_abrv,
                    'name': zone_name,
                    'display': f"{zone_name} ({zone_code})"
                })

            logging.info(f"Loaded {len(self.zones_data)} zones for {len(self.zones_by_state)} states")

        except Exception as e:
            logging.error(f"Error loading zone definitions: {e}")

    def auto_detect_state_from_file(self, input_file):
        """Auto-detect state and zone from point cloud center coordinates."""
        try:
            import laspy
            from pyproj import CRS, Transformer
            from state_bounds import get_state_from_coords

            print(f"DEBUG: Reading file: {input_file}")

            # Read LAS file
            las = laspy.read(input_file)
            header = las.header

            # Get scale and offset from header
            x_scale = header.x_scale
            y_scale = header.y_scale
            x_offset = header.x_offset
            y_offset = header.y_offset

            print(f"DEBUG: Scale: X={x_scale}, Y={y_scale}")
            print(f"DEBUG: Offset: X={x_offset}, Y={y_offset}")

            # Get actual scaled coordinates - handle both laspy versions
            try:
                # Check laspy version and get coordinates appropriately
                if hasattr(las, 'x') and hasattr(las, 'y'):
                    # laspy 2.x - coordinates should already be scaled
                    x_test = las.x
                    y_test = las.y

                    # Test first coordinate to see if scaling is applied
                    test_x = float(x_test[0])

                    # If absurdly large (like -15227280980), it's NOT scaled - manually scale
                    if abs(test_x) > 1e8:
                        print("DEBUG: Coordinates NOT auto-scaled, applying manual scaling")
                        x_coords = (las.X * x_scale) + x_offset
                        y_coords = (las.Y * y_scale) + y_offset
                    else:
                        print("DEBUG: Coordinates already scaled by laspy")
                        x_coords = x_test
                        y_coords = y_test
                else:
                    # laspy 1.x - must manually scale
                    print("DEBUG: Using laspy 1.x - manual scaling required")
                    x_coords = (las.X * x_scale) + x_offset
                    y_coords = (las.Y * y_scale) + y_offset

            except Exception as e:
                print(f"ERROR reading coordinates: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Compute bounds from actual scaled coordinates
            min_x = float(x_coords.min())
            max_x = float(x_coords.max())
            min_y = float(y_coords.min())
            max_y = float(y_coords.max())

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            print(f"DEBUG: Bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
            print(f"DEBUG: Center in file CRS: X={center_x:.2f}, Y={center_y:.2f}")

            # Get CRS from header
            input_crs = None
            try:
                input_crs = header.parse_crs()
                if input_crs:
                    print(f"DEBUG: Parsed CRS: {input_crs.name}")
            except Exception as e:
                print(f"DEBUG: Could not parse CRS from header: {e}")

            # If no CRS in header, try to infer from coordinates
            if not input_crs:
                print("DEBUG: Attempting to infer CRS from coordinate ranges...")
                input_crs = self.infer_crs_from_coordinates(center_x, center_y, min_x, max_x, min_y, max_y)

            if not input_crs:
                print("ERROR: Could not determine CRS")
                return None

            # Transform center point to WGS84
            try:
                transformer = Transformer.from_crs(
                    input_crs,
                    CRS.from_epsg(4326),
                    always_xy=True
                )
                lon, lat = transformer.transform(center_x, center_y)

                print(f"DEBUG: Center in WGS84: Lat={lat:.6f}, Lon={lon:.6f}")

                # Validate coordinates
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    print(f"ERROR: Invalid lat/lon: ({lat}, {lon})")
                    return None

            except Exception as e:
                print(f"ERROR transforming coordinates: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Get state from coordinates
            state = get_state_from_coords(lat, lon)
            print(f"DEBUG: Detected state: {state}")

            # Try to auto-detect zone using zone bounds
            detected_zone = None
            if self.zone_bounds_df is not None and state:
                detected_zone = self.detect_zone_from_bounds(lat, lon, state)
                if detected_zone:
                    print(f"DEBUG: Detected zone: {detected_zone}")

            if state:
                return {
                    'state': state,
                    'lat': lat,
                    'lon': lon,
                    'zone': detected_zone  # May be None
                }

            return None

        except Exception as e:
            print(f"ERROR in auto_detect_state_from_file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def auto_detect_state_from_file(self, input_file):
        """Auto-detect state and zone from point cloud center coordinates."""
        try:
            import laspy
            from pyproj import CRS, Transformer
            from state_bounds import get_state_from_coords

            print(f"DEBUG: Reading file: {input_file}")

            # Read LAS file
            las = laspy.read(input_file)
            header = las.header

            # CRITICAL FIX: Manually scale coordinates using header scale/offset
            # The header min/max might be raw integers, not scaled values

            # Get scale and offset from header
            x_scale = header.x_scale
            y_scale = header.y_scale
            x_offset = header.x_offset
            y_offset = header.y_offset

            print(f"DEBUG: Scale: X={x_scale}, Y={y_scale}")
            print(f"DEBUG: Offset: X={x_offset}, Y={y_offset}")

            # Read actual point coordinates and scale them properly
            # Use a sample of points to compute bounds (faster than all points)
            try:
                # laspy 2.x
                x_coords = las.x
                y_coords = las.y
            except AttributeError:
                # laspy 1.x - manually scale
                x_coords = (las.X * x_scale) + x_offset
                y_coords = (las.Y * y_scale) + y_offset

            # Compute bounds from actual scaled coordinates
            min_x = float(x_coords.min())
            max_x = float(x_coords.max())
            min_y = float(y_coords.min())
            max_y = float(y_coords.max())

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            print(f"DEBUG: Bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
            print(f"DEBUG: Center in file CRS: X={center_x:.2f}, Y={center_y:.2f}")

            # Get CRS from header
            input_crs = None
            try:
                input_crs = header.parse_crs()
                if input_crs:
                    print(f"DEBUG: Parsed CRS: {input_crs.name}")
            except Exception as e:
                print(f"DEBUG: Could not parse CRS from header: {e}")

            # If no CRS in header, try to infer from coordinates
            if not input_crs:
                print("DEBUG: Attempting to infer CRS from coordinate ranges...")
                input_crs = self.infer_crs_from_coordinates(center_x, center_y, min_x, max_x, min_y, max_y)

            if not input_crs:
                print("ERROR: Could not determine CRS")
                return None

            # Transform center point to WGS84
            try:
                transformer = Transformer.from_crs(
                    input_crs,
                    CRS.from_epsg(4326),
                    always_xy=True
                )
                lon, lat = transformer.transform(center_x, center_y)

                print(f"DEBUG: Center in WGS84: Lat={lat:.6f}, Lon={lon:.6f}")

                # Validate coordinates
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    print(f"ERROR: Invalid lat/lon: ({lat}, {lon})")
                    return None

            except Exception as e:
                print(f"ERROR transforming coordinates: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Get state from coordinates
            state = get_state_from_coords(lat, lon)
            print(f"DEBUG: Detected state: {state}")

            # Try to auto-detect zone using zone bounds
            detected_zone = None
            if self.zone_bounds_df is not None and state:
                detected_zone = self.detect_zone_from_bounds(lat, lon, state)
                if detected_zone:
                    print(f"DEBUG: Detected zone: {detected_zone}")

            if state:
                return {
                    'state': state,
                    'lat': lat,
                    'lon': lon,
                    'zone': detected_zone  # May be None
                }

            return None

        except Exception as e:
            print(f"ERROR in auto_detect_state_from_file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def infer_crs_from_coordinates(self, center_x, center_y, min_x, max_x, min_y, max_y):
        """Infer CRS from coordinate ranges using spatial extent."""
        from pyproj import CRS, Transformer

        # Common NAD83(2011) State Plane zones to test
        # Format: EPSG code -> (typical X range, typical Y range, name)
        common_zones = {
            # Oregon
            6558: ((1500000, 3000000), (0, 1500000), "Oregon North"),
            6559: ((1500000, 3000000), (0, 1000000), "Oregon South"),
            6842: ((0, 500000), (0, 1500000), "Oregon Coast"),

            # Washington
            6597: ((1500000, 3000000), (0, 1500000), "Washington North"),
            6598: ((1500000, 3000000), (0, 1000000), "Washington South"),

            # California
            6417: ((1500000, 2500000), (0, 2000000), "California Zone 1"),
            6418: ((1500000, 2500000), (0, 2000000), "California Zone 2"),
            6419: ((1500000, 2500000), (0, 2000000), "California Zone 3"),
            6420: ((1500000, 2500000), (0, 2000000), "California Zone 4"),
            6421: ((1500000, 2500000), (0, 2000000), "California Zone 5"),
            6422: ((1500000, 2500000), (0, 2000000), "California Zone 6"),

            # Alaska
            6394: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 1"),
            6395: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 2"),
            6396: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 3"),
            6397: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 4"),
            6398: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 5"),
            6399: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 6"),
            6400: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 7"),
            6401: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 8"),
            6402: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 9"),
            6403: ((-3000000, 3000000), (-3000000, 3000000), "Alaska Zone 10"),
        }

        # Test against known zones
        for epsg, (x_range, y_range, name) in common_zones.items():
            x_min_range, x_max_range = x_range
            y_min_range, y_max_range = y_range

            # Check if coordinates fall within expected range (with buffer)
            if (x_min_range <= center_x <= x_max_range and y_min_range <= center_y <= y_max_range):
                print(f"DEBUG: Coordinates match EPSG:{epsg} ({name})")
                try:
                    test_crs = CRS.from_epsg(epsg)

                    # Verify by transforming to WGS84
                    transformer = Transformer.from_crs(test_crs, CRS.from_epsg(4326), always_xy=True)
                    lon, lat = transformer.transform(center_x, center_y)

                    # Check if result is reasonable for US (including Alaska and Hawaii)
                    if ((-180 <= lon <= -66 and 24 <= lat <= 72) or  # Continental US + Alaska
                            (144 <= lon <= 146 and 13 <= lat <= 14)):  # Guam
                        print(f"DEBUG: Verified - using EPSG:{epsg}")
                        return test_crs
                except Exception as e:
                    print(f"DEBUG: Failed to verify EPSG:{epsg}: {e}")
                    continue

        print("DEBUG: Could not infer CRS from coordinate ranges")
        return None

    def detect_zone_from_bounds(self, lat, lon, state):
        """Detect SPCS2022 zone from lat/lon using zone bounds."""
        if self.zone_bounds_df is None:
            return None

        try:
            import pandas as pd

            # Print column names for debugging
            print(f"DEBUG: Zone bounds CSV columns: {list(self.zone_bounds_df.columns)}")

            # Convert zone code and abbreviation columns to string type
            zone_code_str = self.zone_bounds_df['Zone code'].astype(str)
            zone_abrv_str = self.zone_bounds_df['Zone abrv'].astype(str)

            # Filter zones that match the state
            state_zones = self.zone_bounds_df[
                zone_code_str.str.startswith(state, na=False) |
                zone_abrv_str.str.startswith(state, na=False)
                ]

            print(f"DEBUG: Found {len(state_zones)} zones for state {state}")

            if len(state_zones) == 0:
                print(f"DEBUG: No zones found for state {state}")
                return None

            # Check each zone's bounds
            for idx, row in state_zones.iterrows():
                # Use actual column names from the CSV
                try:
                    lat_min = row['Min lat (deg)']
                    lat_max = row['Max lat (deg)']

                    # Longitude columns have east/west - use west for negative lon
                    lon_min = row['Min lon west (deg)']
                    lon_max = row['Max lon west (deg)']

                    print(
                        f"DEBUG: Checking zone {row['Zone code']}: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")

                    if pd.notna(lat_min) and pd.notna(lat_max) and pd.notna(lon_min) and pd.notna(lon_max):
                        if (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                            # Return zone abbreviation
                            zone_code = row['Zone abrv']
                            print(f"DEBUG: Matched zone: {zone_code}")
                            return str(zone_code)
                except Exception as e:
                    print(f"DEBUG: Error checking zone {idx}: {e}")
                    continue

            print(f"DEBUG: No zone matched for coordinates ({lat}, {lon})")
            return None

        except Exception as e:
            print(f"ERROR in detect_zone_from_bounds: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_ui(self):
        """Create the user interface with tabs"""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="NSRS Modernization Transformation Tool",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky="w")

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        # Create tabs
        self.transform_tab = ttk.Frame(self.notebook, padding=10)
        self.geoid_tab = ttk.Frame(self.notebook, padding=10)
        self.log_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.transform_tab, text="Transform")
        self.notebook.add(self.geoid_tab, text="Geoid")
        self.notebook.add(self.log_tab, text="Log")

        # Setup each tab
        self.create_transform_tab()
        self.create_geoid_tab()
        self.create_log_tab()

        # Status bar at bottom
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)

    def create_transform_tab(self):
        """Create the Transform tab with collapsible sections"""
        # Make tab scrollable
        canvas = tk.Canvas(self.transform_tab)
        scrollbar = ttk.Scrollbar(self.transform_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        scrollable_frame.columnconfigure(0, weight=1)

        # Sections
        self.create_input_section(scrollable_frame, row=0)
        self.create_output_section(scrollable_frame, row=1)
        self.create_epoch_section(scrollable_frame, row=2)
        self.create_velocity_section(scrollable_frame, row=3)
        self.create_progress_section(scrollable_frame, row=4)
        self.create_control_buttons(scrollable_frame, row=5)

    def create_collapsible_section(self, parent, title, row):
        """Create a collapsible section"""
        container = ttk.Frame(parent)
        container.grid(row=row, column=0, sticky="ew", pady=5)
        container.columnconfigure(0, weight=1)

        # Header with toggle
        header_frame = ttk.Frame(container)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.columnconfigure(1, weight=1)

        self.section_states[title] = tk.BooleanVar(value=True)

        toggle_btn = ttk.Button(
            header_frame,
            text="▼",
            width=3,
            command=lambda: self.toggle_section(title, content_frame, toggle_btn)
        )
        toggle_btn.grid(row=0, column=0, padx=(0, 5))

        title_label = ttk.Label(header_frame, text=title, font=("Arial", 10, "bold"))
        title_label.grid(row=0, column=1, sticky="w")

        # Content frame
        content_frame = ttk.LabelFrame(container, padding=10)
        content_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        content_frame.columnconfigure(1, weight=1)

        return content_frame

    def toggle_section(self, title, frame, button):
        """Toggle section visibility"""
        if self.section_states[title].get():
            frame.grid_remove()
            button.config(text="▶")
            self.section_states[title].set(False)
        else:
            frame.grid()
            button.config(text="▼")
            self.section_states[title].set(True)

    def create_input_section(self, parent, row):
        """Create input file section with batch processing support"""
        input_frame = self.create_collapsible_section(parent, "Input File(s)", row)

        # Buttons row
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))

        ttk.Button(button_frame, text="Add File(s)...", command=self.browse_input_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self.clear_all_files).pack(side=tk.LEFT, padx=5)

        # File list
        ttk.Label(input_frame, text="Selected Files:").grid(row=1, column=0, sticky="nw", padx=(0, 5), pady=(5, 0))

        list_frame = ttk.Frame(input_frame)
        list_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=(5, 0))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(list_frame, height=6, yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)

        # Input CRS
        ttk.Label(input_frame, text="Input CRS:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))

        crs_frame = ttk.Frame(input_frame)
        crs_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=(10, 0))

        crs_entry = ttk.Entry(crs_frame, textvariable=self.input_crs_var, width=20)
        crs_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(crs_frame, text="Auto-detect", command=self.auto_detect_input_crs).pack(side=tk.LEFT, padx=5)

        # Common CRS dropdown - Oregon & Alaska NAD83(2011) Projected Zones
        common_crs = [
            "-- Geographic --",
            "EPSG:6319 - NAD83(2011) Geographic 2D",
            "EPSG:6318 - NAD83(2011) Geographic 3D",
            "",
            "-- Oregon CRS Zones (partial list) --",
            "EPSG:6842 - Oregon (NAD83(2011)) Oregon Coast",
            "EPSG:6558 - Oregon North (NAD83(2011))",
            "EPSG:6560 - Oregon South (NAD83(2011))",
            "EPSG:32610 - UTM 10N WGS84",
            "",
            "-- Alaska Spatial References (partial list) --",
            "EPSG:26931 - NAD83 | Alaska Zone 1",
            "EPSG:26932 - NAD83 | Alaska Zone 2",
            "EPSG:26933 - NAD83 | Alaska Zone 3",
            "EPSG:26934 - NAD83 | Alaska Zone 4",
            "EPSG:26935 - NAD83 | Alaska Zone 5",
            "EPSG:26936 - NAD83 | Alaska Zone 6",
            "EPSG:6397 - NAD83(2011) Alaska Zone 4",
        ]
        crs_combo = ttk.Combobox(crs_frame, values=common_crs, width=50, state="readonly")
        crs_combo.pack(side=tk.LEFT, padx=5)

        def on_crs_select(event):
            """Handle CRS selection, extract EPSG code"""
            selected = crs_combo.get()
            if selected and ' - ' in selected and not selected.startswith('--'):
                epsg = selected.split(' - ')[0].strip()
                self.input_crs_var.set(epsg)

        crs_combo.bind('<<ComboboxSelected>>', on_crs_select)

        # Info
        info_text = (
            "Select input EPSG code. For Oregon data:\n"
            "  • EPSG:6842 (Oregon Coast) - Use if data is coastal Oregon\n"
            "  • EPSG:6558(Oregon North) - Use if data is central/northern Oregon\n"
            "  • EPSG:6560 (Oregon South) - Use if data is in southern Oregon\n"
            "If unsure, try EPSG:32610 first (matches UTM Zone 10N)"
        )
        ttk.Label(
            input_frame,
            text=info_text,
            foreground="gray",
            font=("Arial", 8),
            justify="left"
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

    def create_output_section(self, parent, row):
        """Create output configuration section"""
        output_frame = self.create_collapsible_section(parent, "Output Configuration", row)

        # Output file
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_file_var, width=60)
        output_entry.grid(row=0, column=1, sticky="ew", padx=5)

        browse_out_btn = ttk.Button(output_frame, text="Browse...", command=self.browse_output)
        browse_out_btn.grid(row=0, column=2, padx=5)

        # State selection
        ttk.Label(output_frame, text="State:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))

        state_list = sorted(self.zones_by_state.keys())
        self.state_combo = ttk.Combobox(
            output_frame,
            textvariable=self.state_var,
            values=state_list,
            state="readonly",
            width=20
        )
        self.state_combo.grid(row=1, column=1, sticky="w", padx=5, pady=(10, 0))
        self.state_combo.bind('<<ComboboxSelected>>', self.on_state_selected)

        # Zone selection
        ttk.Label(output_frame, text="SPCS2022 Zone:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))

        self.zone_combo = ttk.Combobox(
            output_frame,
            textvariable=self.zone_var,
            state="readonly",
            width=50
        )
        self.zone_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=(10, 0))
        self.zone_combo.bind('<<ComboboxSelected>>', self.on_zone_selected)

        auto_zone_btn = ttk.Button(output_frame, text="Auto-detect", command=self.auto_detect_zone)
        auto_zone_btn.grid(row=2, column=2, padx=5, pady=(10, 0))

    def add_output_units_selection(self):
        """Add output units selection to the GUI"""

        # Find the zone selection frame or create a new frame after it
        units_frame = ttk.LabelFrame(self.input_frame, text="Output Units", padding="10")
        units_frame.pack(fill=tk.X, padx=10, pady=5)

        # Output units variable
        self.output_units_var = tk.StringVar(value="meters")

        # Radio buttons for unit selection
        ttk.Radiobutton(
            units_frame,
            text="Meters",
            variable=self.output_units_var,
            value="meters"
        ).pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(
            units_frame,
            text="International Feet",
            variable=self.output_units_var,
            value="feet"
        ).pack(side=tk.LEFT, padx=10)

        # Auto-detect checkbox
        self.auto_detect_units = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            units_frame,
            text="Auto-detect from input",
            variable=self.auto_detect_units,
            command=self.on_auto_detect_toggle
        ).pack(side=tk.LEFT, padx=20)

        # Info label
        info_label = ttk.Label(
            units_frame,
            text="Note: SPCS2022 zones support both meters and international feet",
            font=('TkDefaultFont', 8)
        )
        info_label.pack(side=tk.LEFT, padx=10)

    def on_auto_detect_toggle(self):
        """Handle auto-detect checkbox toggle"""
        if self.auto_detect_units.get():
            # Disable manual selection
            for child in self.units_frame.winfo_children():
                if isinstance(child, ttk.Radiobutton):
                    child.configure(state='disabled')
        else:
            # Enable manual selection
            for child in self.units_frame.winfo_children():
                if isinstance(child, ttk.Radiobutton):
                    child.configure(state='normal')

    def on_state_selected(self, event=None):
        """Handle state selection - update zone dropdown"""
        state = self.state_var.get()
        if state in self.zones_by_state:
            zones = self.zones_by_state[state]
            zone_list = [z['display'] for z in zones]
            self.zone_combo['values'] = zone_list

            # Auto-select if only one zone
            if len(zone_list) == 1:
                self.zone_var.set(zone_list[0])
            elif len(zone_list) > 0:
                self.zone_var.set(zone_list[0])

            logging.info(f"Loaded {len(zone_list)} zones for {state}")

    def create_epoch_section(self, parent, row):
        """Create epoch settings section"""
        epoch_frame = self.create_collapsible_section(parent, "Epoch Settings", row)

        ttk.Label(epoch_frame, text="Source Epoch:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        source_entry = ttk.Entry(epoch_frame, textvariable=self.source_epoch_var, width=15)
        source_entry.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(epoch_frame, text="Target Epoch:").grid(row=0, column=2, sticky="w", padx=(20, 5))
        target_entry = ttk.Entry(epoch_frame, textvariable=self.target_epoch_var, width=15)
        target_entry.grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(
            epoch_frame,
            text="Epoch of input data and desired output (e.g., 2010.0, 2020.0)",
            foreground="gray",
            font=("Arial", 9)
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))

    def create_velocity_section(self, parent, row):
        """Create velocity/crustal motion section"""
        vel_frame = self.create_collapsible_section(parent, "Crustal Motion / Velocity Application", row)

        # Method selection
        method_frame = ttk.Frame(vel_frame)
        method_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(method_frame, text="Method:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Radiobutton(
            method_frame,
            text="14-Parameter Helmert Only (Fastest)",
            variable=self.velocity_method_var,
            value="helmert_only",
            command=self.update_velocity_method
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            method_frame,
            text="HTDP Offset Correction (Most Accurate)",
            variable=self.velocity_method_var,
            value="htdp_grid",
            command=self.update_velocity_method
        ).pack(side=tk.LEFT, padx=5)

        # Info
        info_text = (
            "• 14-Parameter Helmert Only: Fast transformation using time-dependent Helmert parameters\n"
            "  - No external files required\n"
            "  - Good accuracy for most applications (~1-2cm)\n\n"
            "• HTDP Offset Correction: Applies HTDP offset to improve accuracy\n"
            "  - Requires HTDP executable (htdp.exe)\n"
            "  - Best accuracy in tectonically active areas (~1mm)\n"
            "  - Single HTDP call for center point + offset applied to all points"
        )
        info_label = ttk.Label(vel_frame, text=info_text, foreground="gray", font=("Arial", 9))
        info_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # HTDP executable path (only for htdp_grid mode)
        ttk.Label(vel_frame, text="HTDP Executable:").grid(row=2, column=0, sticky="w", padx=(0, 5))
        self.velocity_file_entry = ttk.Entry(vel_frame, textvariable=self.velocity_file_var, width=50, state="disabled")
        self.velocity_file_entry.grid(row=2, column=1, sticky="ew", padx=5)

        self.browse_vel_btn = ttk.Button(vel_frame, text="Browse...", command=self.browse_htdp_executable,
                                         state="disabled")
        self.browse_vel_btn.grid(row=2, column=2, padx=5)

        # Status indicator
        ttk.Label(vel_frame, text="Status:").grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.velocity_type_label = ttk.Label(vel_frame, text="(not needed - using Helmert only)", foreground="gray")
        self.velocity_type_label.grid(row=3, column=1, sticky="w", pady=(5, 0))

        vel_frame.columnconfigure(1, weight=1)

    def create_progress_section(self, parent, row):
        """Create progress section"""
        progress_frame = self.create_collapsible_section(parent, "Progress", row)

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.progress_label = ttk.Label(progress_frame, text="Ready to transform")
        self.progress_label.grid(row=1, column=0, sticky="w")

    def create_control_buttons(self, parent, row):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, pady=20)

        self.transform_btn = ttk.Button(
            button_frame,
            text="Start Transformation",
            command=self.start_transformation,
            width=20
        )
        self.transform_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_transformation,
            state=tk.DISABLED,
            width=15
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

    def create_geoid_tab(self):
        """Create the Geoid configuration tab"""
        self.geoid_tab.columnconfigure(0, weight=1)

        # Title
        ttk.Label(
            self.geoid_tab,
            text="Geoid Model Configuration",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 20))

        # Instructions
        instructions = (
            "Upload geoid model files for orthometric height corrections.\n"
            "Supported formats: .bin (binary) or .tif/.tiff (GeoTIFF)\n"
            "Files will be auto-detected in the current directory if available."
        )
        ttk.Label(
            self.geoid_tab,
            text=instructions,
            foreground="gray",
            font=("Arial", 9)
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 20))

        # GEOID22 (NATRF2022)
        geoid22_frame = ttk.LabelFrame(self.geoid_tab, text="GEOID22 (NATRF2022)", padding=10)
        geoid22_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        geoid22_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            geoid22_frame,
            text="Apply GEOID22",
            variable=self.apply_geoid22_var
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        ttk.Label(geoid22_frame, text="File:").grid(row=1, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(geoid22_frame, textvariable=self.geoid22_file_var).grid(row=1, column=1, sticky="ew", padx=5)

        btn_frame22 = ttk.Frame(geoid22_frame)
        btn_frame22.grid(row=1, column=2, padx=5)
        ttk.Button(btn_frame22, text="Browse...", command=self.browse_geoid22).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame22, text="Validate", command=self.validate_geoid22).pack(side=tk.LEFT, padx=2)

        # GEOID18
        geoid18_frame = ttk.LabelFrame(self.geoid_tab, text="GEOID18 (Legacy)", padding=10)
        geoid18_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        geoid18_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            geoid18_frame,
            text="Remove GEOID18",
            variable=self.remove_geoid18_var
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        ttk.Label(geoid18_frame, text="File:").grid(row=1, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(geoid18_frame, textvariable=self.geoid18_file_var).grid(row=1, column=1, sticky="ew", padx=5)

        btn_frame18 = ttk.Frame(geoid18_frame)
        btn_frame18.grid(row=1, column=2, padx=5)
        ttk.Button(btn_frame18, text="Browse...", command=self.browse_geoid18).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame18, text="Validate", command=self.validate_geoid18).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame18, text="View Map", command=self.show_geoid18_map).pack(side=tk.LEFT, padx=2)

        # GEOID12B
        geoid12b_frame = ttk.LabelFrame(self.geoid_tab, text="GEOID12B (Legacy)", padding=10)
        geoid12b_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        geoid12b_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            geoid12b_frame,
            text="Remove GEOID12B",
            variable=self.remove_geoid12b_var
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        ttk.Label(geoid12b_frame, text="File:").grid(row=1, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(geoid12b_frame, textvariable=self.geoid12b_file_var).grid(row=1, column=1, sticky="ew", padx=5)

        btn_frame12b = ttk.Frame(geoid12b_frame)
        btn_frame12b.grid(row=1, column=2, padx=5)
        ttk.Button(btn_frame12b, text="Browse...", command=self.browse_geoid12b).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame12b, text="Validate", command=self.validate_geoid12b).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame12b, text="View Map", command=self.show_geoid12b_map).pack(side=tk.LEFT, padx=2)

        # Auto-load button
        ttk.Button(
            self.geoid_tab,
            text="Load Geoid Grids",
            command=self.load_and_populate_geoid_files
        ).grid(row=5, column=0, columnspan=3, pady=10)

        # Validation display area
        validation_frame = ttk.LabelFrame(self.geoid_tab, text="Geoid File Information", padding=10)
        validation_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=5)
        validation_frame.columnconfigure(0, weight=1)
        validation_frame.rowconfigure(0, weight=1)

        self.geoid_info_text = scrolledtext.ScrolledText(
            validation_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            font=("Courier", 9),
            state='disabled'
        )
        self.geoid_info_text.grid(row=0, column=0, sticky="nsew")

        # Make the validation frame expand
        self.geoid_tab.rowconfigure(6, weight=1)

    def show_geoid18_map(self):
        """Show GEOID18 coverage map"""
        self.show_geoid_map("geoid18_conus_grids_web.png", "GEOID18 Coverage Map", geoid_subdir="geoid18")

    def show_geoid12b_map(self):
        """Show GEOID12B coverage map - check Alaska then CONUS subdirectories"""
        self.show_geoid_map(None, "GEOID12B Coverage Map", geoid_subdir="geoid12b", search_subdirs=["Alaska", "CONUS"])

    def show_geoid_map(self, filename, title, geoid_file_path="", geoid_subdir="", search_subdirs=None):
        """Display geoid coverage map - searches recursively in geoid subdirectories"""
        map_path = None
        search_locations = []

        def search_recursive(directory, target_filename):
            """Recursively search for file in directory and subdirectories"""
            try:
                for item in directory.rglob('*'):
                    if item.is_file() and item.name == target_filename:
                        return item
            except (PermissionError, OSError):
                pass
            return None

        def find_png_in_subdirs(parent_dir, subdirs):
            """Find first PNG file in specific subdirectories"""
            try:
                for subdir in subdirs:
                    subdir_path = parent_dir / subdir
                    if subdir_path.exists() and subdir_path.is_dir():
                        for item in subdir_path.glob('*.png'):
                            if item.is_file():
                                return item
            except (PermissionError, OSError):
                pass
            return None

        # If search_subdirs provided, look there first
        if search_subdirs and geoid_subdir:
            cwd_subdir = Path.cwd() / geoid_subdir
            if cwd_subdir.exists() and cwd_subdir.is_dir():
                map_path = find_png_in_subdirs(cwd_subdir, search_subdirs)
                if map_path:
                    logging.info(f"Found map in {cwd_subdir}: {map_path}")

            if not map_path:
                script_dir = Path(__file__).parent
                script_subdir = script_dir / geoid_subdir
                if script_subdir.exists() and script_subdir.is_dir():
                    map_path = find_png_in_subdirs(script_subdir, search_subdirs)
                    if map_path:
                        logging.info(f"Found map in script subdirectory: {map_path}")

        # If still not found and filename provided, use original search logic
        if not map_path and filename:
            # 1. Try specific geoid subdirectory in cwd (e.g., D:\...\NATRF22_GUI\geoid18)
            if geoid_subdir:
                cwd_subdir = Path.cwd() / geoid_subdir
                search_locations.append(f"{cwd_subdir}")
                if cwd_subdir.exists() and cwd_subdir.is_dir():
                    map_path = search_recursive(cwd_subdir, filename)
                    if map_path:
                        logging.info(f"Found map in cwd subdirectory: {map_path}")

            # 2. Try specific geoid subdirectory in script directory
            if not map_path and geoid_subdir:
                script_dir = Path(__file__).parent
                script_subdir = script_dir / geoid_subdir
                search_locations.append(f"{script_subdir}")
                if script_subdir.exists() and script_subdir.is_dir():
                    map_path = search_recursive(script_subdir, filename)
                    if map_path:
                        logging.info(f"Found map in script subdirectory: {map_path}")

            # 3. Try geoid file directory and subdirectories
            if not map_path and geoid_file_path and os.path.exists(geoid_file_path):
                geoid_dir = Path(geoid_file_path).parent
                search_locations.append(f"{geoid_dir} (and subdirs)")
                map_path = search_recursive(geoid_dir, filename)
                if map_path:
                    logging.info(f"Found map in geoid file directory: {map_path}")

            # 4. Try current working directory
            if not map_path:
                cwd = Path.cwd()
                search_locations.append(f"{cwd}")
                map_path = search_recursive(cwd, filename)
                if map_path:
                    logging.info(f"Found map in cwd: {map_path}")

        if not map_path:
            search_desc = f"searched subdirs: {search_subdirs}" if search_subdirs else f"searched for: {filename}"
            locations_str = "\n".join(f"  • {loc}" for loc in search_locations[:5])
            messagebox.showinfo(
                "Map Not Found",
                f"Coverage map not found ({search_desc}).\n\nSearched in:\n{locations_str}"
            )
            return

        try:
            map_window = tk.Toplevel(self.root)
            map_window.title(title)
            img = Image.open(map_path)
            max_size = (1200, 800)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(map_window, image=photo)
            label.image = photo
            label.pack()
            ttk.Button(map_window, text="Close", command=map_window.destroy).pack(pady=10)
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying map: {str(e)}")

    def create_log_tab(self):
        """Create the Log tab"""
        self.log_tab.columnconfigure(0, weight=1)
        self.log_tab.rowconfigure(0, weight=1)

        # Log text widget
        self.log_text = scrolledtext.ScrolledText(
            self.log_tab,
            wrap=tk.WORD,
            width=100,
            height=30,
            font=("Courier", 9),
            state='disabled'
        )
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Clear log button
        clear_btn = ttk.Button(self.log_tab, text="Clear Log", command=self.clear_log)
        clear_btn.grid(row=1, column=0, pady=5)

        # Setup log handler
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)

    def clear_log(self):
        """Clear the log display"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

    def browse_input_file(self):
        """Browse for input LAZ/LAS file(s) - supports multiple selection"""
        filenames = filedialog.askopenfilenames(
            title="Select Input Point Cloud(s)",
            filetypes=[
                ("Point Cloud files", "*.laz *.las"),
                ("All files", "*.*")
            ]
        )

        if filenames:
            # Add files to the list
            for filename in filenames:
                if filename not in self.input_files:
                    self.input_files.append(filename)
                    self.file_listbox.insert(tk.END, os.path.basename(filename))

            # Update input_file_var with first file for compatibility
            if self.input_files:
                self.input_file_var.set(self.input_files[0])

            # Auto-detect state and zone from first file
            if filenames:
                self.status_var.set("Detecting location from point cloud...")
                self.root.update_idletasks()

                try:
                    detection = self.auto_detect_state_from_file(filenames[0])

                    if detection:
                        state = detection['state']
                        lat = detection['lat']
                        lon = detection['lon']
                        zone = detection.get('zone', None)

                        # Set state dropdown
                        self.state_var.set(state)
                        self.on_state_selected(None)  # Filter zones for this state

                        # If zone detected, try to select it
                        if zone:
                            # Get values from the combobox widget, not the StringVar
                            zone_values = self.zone_combo['values']  # CHANGED from self.zone_var['values']

                            # Find matching zone in dropdown
                            for zone_display in zone_values:
                                if f"({zone})" in zone_display:
                                    self.zone_var.set(zone_display)
                                    status_msg = f"✓ Auto-detected: {state} - {zone} (Lat: {lat:.4f}°, Lon: {lon:.4f}°)"
                                    break
                            else:
                                status_msg = f"✓ Auto-detected: {state} (Lat: {lat:.4f}°, Lon: {lon:.4f}°) - Select zone"
                        else:
                            status_msg = f"✓ Auto-detected: {state} (Lat: {lat:.4f}°, Lon: {lon:.4f}°) - Select zone"

                        self.status_var.set(status_msg)
                        self.auto_generate_output_filename()

                    else:
                        self.status_var.set("⚠ Could not detect location - Please select manually")

                except Exception as e:
                    print(f"Error during detection: {e}")
                    import traceback
                    traceback.print_exc()
                    self.status_var.set("⚠ Error detecting location - Please select manually")

    def auto_generate_output_filename(self):
        """Auto-generate output filename based on input file and selected zone."""
        try:
            input_file = self.input_file_var.get()
            zone_display = self.zone_var.get()

            if not input_file or not zone_display:
                return

            # Extract base name and extension
            input_dir = os.path.dirname(input_file)
            input_basename = os.path.basename(input_file)
            input_name, input_ext = os.path.splitext(input_basename)

            # Extract zone abbreviation from "Zone Name (ABBR)" format
            import re
            match = re.search(r'\(([A-Z0-9]+)\)', zone_display)
            if match:
                zone_abbr = match.group(1)
            else:
                # If no abbreviation found, use the whole zone string
                zone_abbr = zone_display.replace(" ", "_")

            # Generate output filename: inputname_SPCS2022_ZONE.ext
            output_filename = f"{input_name}_SPCS2022_{zone_abbr}{input_ext}"
            output_path = os.path.join(input_dir, output_filename)

            # Set the output file variable
            self.output_file_var.set(output_path)

            print(f"DEBUG: Auto-generated output filename: {output_filename}")

        except Exception as e:
            print(f"ERROR in auto_generate_output_filename: {e}")
            import traceback
            traceback.print_exc()

    def remove_selected_file(self):
        """Remove selected file(s) from the list"""
        selected_indices = self.file_listbox.curselection()
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.input_files[index]

        # Update input_file_var
        if self.input_files:
            self.input_file_var.set(self.input_files[0])
        else:
            self.input_file_var.set("")

    def clear_all_files(self):
        """Clear all files from the list"""
        self.file_listbox.delete(0, tk.END)
        self.input_files = []
        self.input_file_var.set("")

    def browse_output(self):
        """Browse for output file"""
        filename = filedialog.asksaveasfilename(
            title="Save Output LAS File",
            defaultextension=".laz",
            filetypes=[("LAZ Files", "*.laz"), ("LAS Files", "*.las"), ("All Files", "*.*")]
        )
        if filename:
            self.output_file_var.set(filename)

    def browse_velocity_file(self):
        """Browse for velocity file"""
        method = self.velocity_method_var.get()

        if method == "htdp_grid":
            filetypes = [("HTDP Fortran", "*.f"), ("All Files", "*.*")]
        elif method == "cors_idw":
            filetypes = [("CORS Text", "*.txt"), ("All Files", "*.*")]
        else:
            filetypes = [("All Files", "*.*")]

        filename = filedialog.askopenfilename(title="Select Velocity File", filetypes=filetypes)
        if filename:
            self.velocity_file_var.set(filename)
            self.detect_velocity_file_type(filename)

    def browse_geoid22(self):
        """Browse for GEOID22 file"""
        filename = filedialog.askopenfilename(
            title="Select GEOID22 File",
            filetypes=[("Geoid Files", "*.bin *.tif *.tiff"), ("All Files", "*.*")]
        )
        if filename:
            self.geoid22_file_var.set(filename)

    def browse_geoid18(self):
        """Browse for GEOID18 file"""
        filename = filedialog.askopenfilename(
            title="Select GEOID18 File",
            filetypes=[("Geoid Files", "*.bin *.tif *.tiff"), ("All Files", "*.*")]
        )
        if filename:
            self.geoid18_file_var.set(filename)

    def browse_geoid12b(self):
        """Browse for GEOID12B file"""
        filename = filedialog.askopenfilename(
            title="Select GEOID12B File",
            filetypes=[("Geoid Files", "*.bin *.tif *.tiff"), ("All Files", "*.*")]
        )
        if filename:
            self.geoid12b_file_var.set(filename)

    def validate_geoid22(self):
        """Validate GEOID22 file and display info"""
        self.validate_geoid_file(self.geoid22_file_var.get(), "GEOID22")

    def validate_geoid18(self):
        """Validate GEOID18 file and display info"""
        self.validate_geoid_file(self.geoid18_file_var.get(), "GEOID18")

    def validate_geoid12b(self):
        """Validate GEOID12B file and display info"""
        self.validate_geoid_file(self.geoid12b_file_var.get(), "GEOID12B")

    def validate_geoid_file(self, filepath, name):
        """Validate and display geoid file information"""
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", f"{name} file not found")
            return

        try:
            from geoid_utils import NGSGeoidBin

            geoid = NGSGeoidBin(filepath)

            info = f"\n{'=' * 60}\n"
            info += f"{name} File Information\n"
            info += f"{'=' * 60}\n"
            info += f"File: {os.path.basename(filepath)}\n"
            info += f"Path: {filepath}\n\n"
            info += f"Grid Parameters:\n"
            info += f"  South Latitude:  {geoid.header.slat:.4f}°\n"
            info += f"  West Longitude:  {geoid.header.wlon:.4f}°\n"
            info += f"  Delta Latitude:  {geoid.header.dlat:.6f}°\n"
            info += f"  Delta Longitude: {geoid.header.dlon:.6f}°\n"
            info += f"  Grid Size (lat): {geoid.header.nlat}\n"
            info += f"  Grid Size (lon): {geoid.header.nlon}\n"
            info += f"  Total Points:    {geoid.header.nlat * geoid.header.nlon:,}\n"
            info += f"  IKind:           {geoid.header.ikind}\n\n"

            # Calculate coverage
            north_lat = geoid.header.slat + (geoid.header.nlat - 1) * geoid.header.dlat
            east_lon = geoid.header.wlon + (geoid.header.nlon - 1) * geoid.header.dlon
            info += f"Coverage:\n"
            info += f"  Latitude:  {geoid.header.slat:.2f}° to {north_lat:.2f}°\n"
            info += f"  Longitude: {geoid.header.wlon:.2f}° to {east_lon:.2f}°\n"
            info += f"\n✓ File validated successfully\n"

            self.geoid_info_text.configure(state='normal')
            self.geoid_info_text.insert(tk.END, info)
            self.geoid_info_text.configure(state='disabled')
            self.geoid_info_text.see(tk.END)

            logging.info(f"Validated {name}: {os.path.basename(filepath)}")

        except Exception as e:
            error_msg = f"\n{'=' * 60}\n"
            error_msg += f"Error validating {name}\n"
            error_msg += f"{'=' * 60}\n"
            error_msg += f"File: {filepath}\n"
            error_msg += f"Error: {str(e)}\n"

            self.geoid_info_text.configure(state='normal')
            self.geoid_info_text.insert(tk.END, error_msg)
            self.geoid_info_text.configure(state='disabled')
            self.geoid_info_text.see(tk.END)

            messagebox.showerror("Validation Error", f"Error validating {name}:\n{str(e)}")

    # Auto-detection methods
    def auto_detect_input(self):
        """Auto-detect input LAS file"""
        cwd = Path.cwd()
        las_files = list(cwd.glob("*.las")) + list(cwd.glob("*.laz"))

        if las_files:
            self.input_file_var.set(str(las_files[0]))
            self.auto_suggest_output()
            self.auto_detect_zone()
            self.status_var.set(f"Auto-detected input: {las_files[0].name}")

    def auto_detect_input_crs(self):
        """Auto-detect input CRS from LAS file"""
        # Use first file in the list
        if not self.input_files:
            messagebox.showinfo("Info", "Please add at least one input file first")
            return

        input_file = self.input_files[0]

        try:
            import laspy
            las = laspy.read(input_file)

            # Try to get CRS from VLRs
            if hasattr(las, 'header') and hasattr(las.header, 'vlrs'):
                for vlr in las.header.vlrs:
                    if vlr.record_id == 2112:  # WKT CRS
                        # Parse WKT to get EPSG
                        logging.info("Found WKT CRS in file")

            # Check coordinate ranges
            x_min, x_max = float(np.min(las.x)), float(np.max(las.x))
            y_min, y_max = float(np.min(las.y)), float(np.max(las.y))

            logging.info(f"Coordinate ranges: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")

            # Heuristic detection
            if -180 <= x_min <= 180 and -90 <= y_min <= 90:
                self.input_crs_var.set("EPSG:6319")
                messagebox.showinfo("Auto-detect", "Detected geographic coordinates (EPSG:6319)")
            else:
                messagebox.showinfo("Auto-detect", "Could not determine CRS. Please specify manually.")

        except Exception as e:
            messagebox.showerror("Error", f"Error detecting CRS:\n{str(e)}")

    # GUI_transformation_tool.py

    # Add these methods after the auto_suggest_output method (around line 1069)

    def on_zone_selected(self, event=None):
        """Update output filename when zone is selected"""
        self.update_output_with_zone()

    def update_output_with_zone(self):
        """Update output filename to include zone abbreviation"""
        input_file = self.input_file_var.get()
        zone_display = self.zone_var.get()

        if input_file and zone_display:
            # Extract zone code from display
            try:
                zone_code = zone_display.split("(")[1].rstrip(")")

                # Find zone abbreviation
                zone_abrv = None
                for zone in self.zones_data:
                    if zone.get('Zone Code', '').strip() == zone_code:
                        zone_abrv = zone.get('Zone Abrv', '').strip()
                        break

                if zone_abrv:
                    input_path = Path(input_file)
                    output_name = f"{input_path.stem}_SPCS2022_{zone_abrv}.laz"
                    output_path = input_path.parent / output_name
                    self.output_file_var.set(str(output_path))
            except:
                # If parsing fails, just use default naming
                self.auto_suggest_output()

    def auto_suggest_output(self):
        """Auto-suggest output filename"""
        input_file = self.input_file_var.get()
        if input_file:
            input_path = Path(input_file)
            output_name = input_path.stem + "_SPCS2022.laz"
            output_path = input_path.parent / output_name
            self.output_file_var.set(str(output_path))

    def detect_input_units(self, input_file):
        """
        Detect and display the likely units of the input file
        """
        import laspy

        try:
            with laspy.open(input_file) as las_file:
                chunk = next(las_file.chunk_iterator(1000))
                x_range = chunk.x.max() - chunk.x.min()
                y_range = chunk.y.max() - chunk.y.min()
                avg_range = (x_range + y_range) / 2

                if avg_range > 10000:
                    detected = "feet (large coordinates)"
                elif avg_range < 1000:
                    detected = "meters or degrees"
                else:
                    detected = "uncertain"

                # Update status or info label
                self.status_var.set(f"Input appears to be in {detected}")

                # If auto-detect is on, set the output units
                if hasattr(self, 'auto_detect_units') and self.auto_detect_units.get():
                    if "feet" in detected:
                        self.output_units_var.set("feet")
                    else:
                        self.output_units_var.set("meters")

        except Exception as e:
            logging.warning(f"Could not detect input units: {e}")

    def auto_detect_zone(self):
        """Auto-detect SPCS zone from input file"""
        input_file = self.input_file_var.get()
        if not input_file or not os.path.exists(input_file):
            return

        try:
            import laspy
            las = laspy.read(input_file)

            # Fix: Convert to numpy array before calling mean()
            lat_center = float(np.mean(np.asarray(las.y)))
            lon_center = float(np.mean(np.asarray(las.x)))

            # Detect state first
            state_abbr = self.detect_state_from_coords(lat_center, lon_center)
            if state_abbr:
                self.state_var.set(state_abbr)
                self.on_state_selected()

                # Try to detect specific zone
                zone_display = self.detect_zone_from_coords(lat_center, lon_center, state_abbr)
                if zone_display:
                    self.zone_var.set(zone_display)
                    self.status_var.set(f"Auto-detected: {state_abbr} - {zone_display}")
                else:
                    self.status_var.set(f"Auto-detected state: {state_abbr}")

        except Exception as e:
            logging.warning(f"Could not auto-detect zone: {e}")

    def detect_state_from_coords(self, lat, lon):
        """Detect state from coordinates"""
        if lat > 54.0:
            return "AK"  # Alaska
        elif 24.0 < lat < 50.0 and -125.0 < lon < -66.0:
            # CONUS detection by longitude ranges and latitude
            if lon < -120.0:  # Western states
                if lat > 48.0:
                    return "WA"
                elif lat > 42.0:
                    return "OR"
                elif lat > 39.0:
                    return "CA"
                elif lat > 37.0:
                    return "NV"
                else:
                    return "NV"
            elif lon < -110.0:  # Mountain states
                if lat > 45.0:
                    return "ID"
                elif lat > 43.0:
                    return "UT"
                elif lat > 40.0:
                    return "CO"
                else:
                    return "NM"
            elif lon < -100.0:  # Central states
                if lat > 49.0:
                    return "MT"
                elif lat > 45.0:
                    return "WY"
                elif lat > 42.0:
                    return "NE"
                else:
                    return "KS"
            elif lon < -90.0:  # Midwest
                if lat > 46.0:
                    return "MN"
                elif lat > 44.0:
                    return "WI"
                elif lat > 42.0:
                    return "MI"
                elif lat > 40.0:
                    return "IL"
                else:
                    return "MO"
            elif lon < -80.0:  # Eastern states
                if lat > 44.0:
                    return "NY"
                elif lat > 42.0:
                    return "PA"
                elif lat > 40.0:
                    return "VA"
                else:
                    return "NC"
            else:
                return "ME"

        return None

    def detect_zone_from_coords(self, lat, lon, state_abbr):
        """Detect specific zone from coordinates within a state"""
        if state_abbr not in self.zones_by_state:
            return None

        zones = self.zones_by_state[state_abbr]

        if len(zones) == 1:
            return zones[0]['display']

        # Simple heuristic for multi-zone states
        if state_abbr == "AK":
            if lon < -164.0:
                return self.find_zone_display(state_abbr, "AK_SE")
            elif lon < -156.0:
                return self.find_zone_display(state_abbr, "AK_GP")
            elif lon < -148.0:
                return self.find_zone_display(state_abbr, "AK_AN")

        return None

    def find_zone_display(self, state_abbr, zone_abrv):
        """Find zone display string by abbreviation"""
        if state_abbr in self.zones_by_state:
            for zone in self.zones_by_state[state_abbr]:
                if zone['abrv'] == zone_abrv:
                    return zone['display']
        return None

    def load_and_populate_geoid_files(self):
        """Load geoid files and populate file information without requiring Validate button"""
        self.auto_detect_geoid_files()

        # Clear existing info
        self.geoid_info_text.configure(state='normal')
        self.geoid_info_text.delete(1.0, tk.END)
        self.geoid_info_text.configure(state='disabled')

        # Validate and display each file
        if self.geoid22_file_var.get():
            self.validate_geoid_file(self.geoid22_file_var.get(), "GEOID22")

        if self.geoid18_file_var.get():
            self.validate_geoid_file(self.geoid18_file_var.get(), "GEOID18")

        if self.geoid12b_file_var.get():
            self.validate_geoid_file(self.geoid12b_file_var.get(), "GEOID12B")

    def auto_detect_geoid_files(self):
        """Auto-detect geoid files in current directory"""
        cwd = Path.cwd()

        # Look for GEOID22
        geoid22_patterns = ["*geoid22*.bin", "*geoid22*.tif", "*g2022*.bin", "*g2022*.tif"]
        for pattern in geoid22_patterns:
            files = list(cwd.glob(pattern))
            if files:
                self.geoid22_file_var.set(str(files[0]))
                logging.info(f"Auto-detected GEOID22: {files[0].name}")
                break

        # Look for GEOID18
        geoid18_patterns = ["*geoid18*.bin", "*geoid18*.tif", "*g2018*.bin", "*g2018*.tif"]
        for pattern in geoid18_patterns:
            files = list(cwd.glob(pattern))
            if files:
                self.geoid18_file_var.set(str(files[0]))
                logging.info(f"Auto-detected GEOID18: {files[0].name}")
                break

        # Look for GEOID12B
        geoid12b_patterns = ["*geoid12b*.bin", "*geoid12b*.tif", "*g2012b*.bin", "*g2012b*.tif"]
        for pattern in geoid12b_patterns:
            files = list(cwd.glob(pattern))
            if files:
                self.geoid12b_file_var.set(str(files[0]))
                logging.info(f"Auto-detected GEOID12B: {files[0].name}")
                break

    def detect_velocity_file_type(self, filename):
        """Auto-detect velocity file type"""
        try:
            with open(filename, 'r') as f:
                first_lines = []
                for _ in range(5):
                    try:
                        first_lines.append(next(f).strip())
                    except StopIteration:
                        break

            if not first_lines:
                self.velocity_type_label.config(text="Empty file", foreground="red")
                return "error"

            # Check for CORS format
            first_data = first_lines[0].split()
            if len(first_data) >= 8:
                try:
                    float(first_data[1])
                    float(first_data[2])
                    float(first_data[5])
                    self.velocity_type_label.config(text="CORS Station File", foreground="blue")
                    return "cors"
                except ValueError:
                    pass

            # Check for HTDP format
            content = ' '.join(first_lines).upper()
            if 'DATA' in content or 'DIMENSION' in content:
                self.velocity_type_label.config(text="HTDP Velocity Grid", foreground="blue")
                return "htdp"

            self.velocity_type_label.config(text="Unknown format", foreground="orange")
            return "unknown"

        except Exception as e:
            self.velocity_type_label.config(text=f"Error: {str(e)}", foreground="red")
            return "error"

    def update_velocity_method(self):
        """Update UI based on selected velocity method"""
        method = self.velocity_method_var.get()

        if method == "helmert_only":
            # Disable file browser - not needed
            self.velocity_file_entry.config(state="disabled")
            self.browse_vel_btn.config(state="disabled")
            self.velocity_type_label.config(
                text="(not needed - using 14-parameter Helmert only)",
                foreground="gray"
            )
            self.velocity_file_var.set("")

        elif method == "htdp_grid":
            # Enable file browser for HTDP executable
            self.velocity_file_entry.config(state="normal")
            self.browse_vel_btn.config(state="normal")
            self.velocity_type_label.config(
                text="Select htdp.exe executable",
                foreground="orange"
            )

            # Try to auto-detect htdp.exe in common locations
            common_paths = [
                r"D:\Winter2025\SPCS2022\20251125_NATRF22_GUI\SPCS22_GUI\HTDP-main\HTDP-main\htdp.exe",
                r"C:\lidar\HTDP-main\htdp.exe",
                r".\HTDP-main\htdp.exe"
            ]

            for path in common_paths:
                if os.path.exists(path):
                    self.velocity_file_var.set(path)
                    self.velocity_type_label.config(
                        text="HTDP executable found",
                        foreground="green"
                    )
                    break

    def browse_htdp_executable(self):
        """Browse for HTDP executable"""
        filename = filedialog.askopenfilename(
            title="Select HTDP Executable",
            filetypes=[
                ("Executable files", "*.exe"),
                ("All files", "*.*")
            ]
        )

        if filename:
            self.velocity_file_var.set(filename)

            # Validate it's htdp.exe
            if os.path.basename(filename).lower() == "htdp.exe":
                self.velocity_type_label.config(
                    text="HTDP executable selected",
                    foreground="green"
                )
            else:
                self.velocity_type_label.config(
                    text="Warning: Expected htdp.exe",
                    foreground="orange"
                )

    def validate_velocity_settings(self):
        """Validate velocity-related settings before transformation"""
        method = self.velocity_method_var.get()

        if method == "helmert_only":
            # No validation needed - Helmert only
            return True

        elif method == "htdp_grid":
            # Check for HTDP executable
            htdp_path = self.velocity_file_var.get().strip()

            if not htdp_path:
                messagebox.showerror(
                    "Missing HTDP Executable",
                    "HTDP Grid mode requires the HTDP executable (htdp.exe).\n\n"
                    "Please browse for htdp.exe or switch to 'Helmert Only' mode."
                )
                return False

            if not os.path.exists(htdp_path):
                messagebox.showerror(
                    "HTDP Executable Not Found",
                    f"Cannot find HTDP executable at:\n{htdp_path}\n\n"
                    "Please verify the path or browse for htdp.exe"
                )
                return False

            if not htdp_path.lower().endswith('.exe'):
                response = messagebox.askyesno(
                    "Unexpected File Type",
                    f"Expected an executable (.exe) file but got:\n{os.path.basename(htdp_path)}\n\n"
                    "Continue anyway?",
                    icon='warning'
                )
                if not response:
                    return False

            return True

        else:
            messagebox.showerror(
                "Invalid Method",
                f"Unknown velocity method: {method}\n\n"
                "Please select either 'Helmert Only' or 'HTDP Offset Correction'"
            )
            return False

    def start_transformation(self):
        """Start the transformation with batch processing support"""
        # Validate inputs
        zone_display = self.zone_var.get()
        input_crs = self.input_crs_var.get()

        if not self.input_files:
            messagebox.showerror("Error", "Please add at least one input file")
            return

        # Validate all files exist
        for input_file in self.input_files:
            if not os.path.exists(input_file):
                messagebox.showerror("Error", f"Input file not found: {os.path.basename(input_file)}")
                return

        if not zone_display:
            messagebox.showerror("Error", "Please select a target zone")
            return

        if not zone_display:
            messagebox.showerror("Error", "Please select a target zone")
            return

        if not input_crs:
            messagebox.showerror("Error", "Please specify input CRS")
            return

        # Extract zone code and find full zone data
        zone_code = zone_display.split("(")[1].rstrip(")")

        # Find the full zone data from zones_data
        zone_info = None
        for zone in self.zones_data:
            if zone.get('Zone Code', '').strip() == zone_code:
                zone_info = zone
                break

        if not zone_info:
            messagebox.showerror("Error", f"Could not find zone data for {zone_code}")
            return

        # Get epochs
        try:
            source_epoch = float(self.source_epoch_var.get())
            target_epoch = float(self.target_epoch_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid epoch values")
            return

        # Build velocity parameters
        velocity_method = self.velocity_method_var.get()
        velocity_params = {
            'velocity_method': velocity_method,
            'source_epoch': source_epoch,
            'target_epoch': target_epoch
        }

        # Validate and add HTDP executable path if using htdp_grid mode
        if velocity_method == "htdp_grid":
            htdp_exe_path = self.velocity_file_var.get().strip()

            if not htdp_exe_path:
                messagebox.showerror(
                    "Missing HTDP Executable",
                    "HTDP Grid mode requires the HTDP executable (htdp.exe).\n\n"
                    "Please browse for htdp.exe or switch to 'Helmert Only' mode."
                )
                return

            if not os.path.exists(htdp_exe_path):
                messagebox.showerror(
                    "HTDP Executable Not Found",
                    f"Cannot find HTDP executable at:\n{htdp_exe_path}\n\n"
                    "Please verify the path or browse for htdp.exe"
                )
                return

            if not htdp_exe_path.lower().endswith('.exe'):
                response = messagebox.askyesno(
                    "Unexpected File Type",
                    f"Expected an executable (.exe) file but got:\n{os.path.basename(htdp_exe_path)}\n\n"
                    "Continue anyway?",
                    icon='warning'
                )
                if not response:
                    return

            velocity_params['htdp_exe_path'] = htdp_exe_path

        # helmert_only mode needs no additional parameters        # Get zone code for batch processing
        zone_code = zone_display.split('(')[1].rstrip(')')

        # Build geoid settings
        geoid_settings = {
            'geoid22_file': self.geoid22_file_var.get(),
            'geoid18_file': self.geoid18_file_var.get(),
            'geoid12b_file': self.geoid12b_file_var.get(),
            'remove_geoid12b': self.remove_geoid12b_var.get(),
            'remove_geoid18': self.remove_geoid18_var.get(),
            'apply_geoid22': self.apply_geoid22_var.get()
        }

        # Update UI
        self.transform_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="Starting transformation...")
        self.status_var.set("Processing...")

        # Switch to log tab
        self.notebook.select(self.log_tab)

        # Run in thread
        def run_transformation():
            import traceback

            if hasattr(self, 'auto_detect_units') and self.auto_detect_units.get():
                output_units = None  # Will auto-detect
            elif hasattr(self, 'output_units_var'):
                output_units = self.output_units_var.get()
            else:
                output_units = "meters"  # Default

            total_files = len(self.input_files)
            successful = 0
            failed = 0
            errors = []

            batch_start_time = time.time()
            try:
                for idx, input_file in enumerate(self.input_files, 1):
                    # Update progress
                    progress = int((idx - 1) / total_files * 100)
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda file_num=idx, total=total_files:
                    self.progress_label.config(text=f"Processing file {file_num}/{total}..."))

                    # Generate output filename
                    input_dir = os.path.dirname(input_file)
                    input_basename = os.path.splitext(os.path.basename(input_file))[0]
                    zone_abrv = zone_info.get('State Abbreviation', 'SPCS')
                    output_file = os.path.join(input_dir, f"{input_basename}_SPCS2022_{zone_code}_{zone_abrv}.laz")

                    try:
                        run_enhanced_transformation(
                            gui=self,
                            velocity_params=velocity_params,
                            input_file=input_file,
                            output_file=output_file,
                            input_crs=input_crs,
                            zone_info=zone_info,
                            geoid_settings=geoid_settings,
                            output_units=output_units
                        )
                        successful += 1
                    except Exception as e:
                        failed += 1
                        error_msg = f"{os.path.basename(input_file)}: {str(e)}"
                        errors.append(error_msg)
                        logging.error(f"Error processing {input_file}:\n{traceback.format_exc()}")

                # Calculate and log batch timing
                batch_time = time.time() - batch_start_time
                logging.info("=" * 80)
                logging.info("BATCH PROCESSING SUMMARY")
                logging.info("=" * 80)
                logging.info(f"Total files processed: {successful + failed}")
                logging.info(f"Successful: {successful}")
                logging.info(f"Failed: {failed}")
                logging.info(f"Total batch processing time: {batch_time:.2f} seconds")
                logging.info(f"Average time per file: {batch_time / (successful + failed):.2f} seconds")
                logging.info("=" * 80)
                # Final status
                if failed == 0:
                    self.root.after(0, self.transformation_complete,
                                    f"Successfully processed {successful}/{total_files} files in {batch_time:.2f} seconds\n\nAverage: {batch_time / (successful + failed):.2f} seconds per file")
                else:
                    error_summary = f"Processed {successful}/{total_files} files successfully in {batch_time:.2f} seconds.\n\nErrors ({failed}):\n" + "\n".join(
                        errors[:5])
                    if len(errors) > 5:
                        error_summary += f"\n... and {len(errors) - 5} more"
                    self.root.after(0, self.transformation_error, error_summary)

            except Exception as e:
                error_details = traceback.format_exc()
                logging.error(f"Transformation error:\n{error_details}")
                self.root.after(0, self.transformation_error, str(e))

        thread = threading.Thread(target=run_transformation, daemon=True)
        thread.start()

    def transformation_complete(self, message=None):
        """Handle successful transformation"""
        self.progress_var.set(100)
        self.progress_label.config(text="Transformation complete!")
        self.status_var.set("Success!")

        self.transform_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

        if message is None:
            message = "Transformation completed successfully!"
        messagebox.showinfo("Success", message)

    def transformation_error(self, error_msg):
        """Handle transformation error"""
        self.progress_label.config(text="Error occurred")
        self.status_var.set("Error")

        self.transform_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

        messagebox.showerror("Transformation Error", f"An error occurred:\n\n{error_msg}")

    def cancel_transformation(self):
        """Cancel ongoing transformation"""
        self.status_var.set("Cancelling...")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = TransformationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()