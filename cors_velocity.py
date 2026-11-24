#!/usr/bin/env python3
"""
CORS Station Velocity Interpolation

Uses actual CORS station velocities to improve accuracy in regions
with poor velocity grid coverage (e.g., Gulf Coast).
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CORSStation:
    """CORS station with position and velocity"""
    site: str
    epoch: float
    x: float  # meters
    y: float  # meters
    z: float  # meters
    vx: float  # mm/yr
    vy: float  # mm/yr
    vz: float  # mm/yr
    country: str
    state: str
    status: str

    # Will be computed
    lat: float = 0.0
    lon: float = 0.0
    h: float = 0.0


def xyz_to_geodetic(x, y, z):
    """Convert ECEF XYZ to geodetic lat/lon/height (WGS84)"""
    a = 6378137.0  # WGS84 semi-major axis
    f = 1.0 / 298.257223563  # WGS84 flattening
    e2 = 2 * f - f * f  # First eccentricity squared

    # Longitude
    lon = math.atan2(y, x)

    # Iterative calculation for latitude and height
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - e2))

    for _ in range(5):  # Usually converges in 2-3 iterations
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - e2 * N / (N + h)))

    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), h


def parse_cors_velocity_file(filename: str) -> List[CORSStation]:
    """
    Parse CORS velocity file

    Format (no header):
    SITE EPOCH x y z Vx Vy Vz Ctry State Status
    """
    stations = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 11:
                continue

            try:
                station = CORSStation(
                    site=parts[0],
                    epoch=float(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    vx=float(parts[5]),
                    vy=float(parts[6]),
                    vz=float(parts[7]),
                    country=parts[8],
                    state=parts[9],
                    status=' '.join(parts[10:])  # Status might be multiple words
                )

                # Compute geodetic coordinates
                station.lat, station.lon, station.h = xyz_to_geodetic(
                    station.x, station.y, station.z
                )

                stations.append(station)

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line[:50]}... ({e})")
                continue

    return stations


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points in kilometers
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class CORSVelocityInterpolator:
    """Interpolate velocities from nearby CORS stations"""

    def __init__(self, stations: List[CORSStation], max_distance_km: float = 500.0):
        """
        Initialize with CORS stations

        Args:
            stations: List of CORS stations
            max_distance_km: Maximum distance to consider for interpolation
        """
        self.stations = stations
        self.max_distance_km = max_distance_km
        print(f"Loaded {len(stations)} CORS stations")

    def get_velocity_xyz(self, lat: float, lon: float,
                         n_neighbors: int = 4,
                         min_neighbors: int = 2) -> Optional[Tuple[float, float, float]]:
        """
        Get interpolated XYZ velocity at a point using inverse distance weighting

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            n_neighbors: Number of nearest neighbors to use
            min_neighbors: Minimum number of neighbors required

        Returns:
            (vx, vy, vz) in mm/yr, or None if insufficient data
        """
        # Find nearby stations
        distances = []
        for station in self.stations:
            dist = haversine_distance(lat, lon, station.lat, station.lon)
            if dist <= self.max_distance_km:
                distances.append((dist, station))

        if len(distances) < min_neighbors:
            return None

        # Sort by distance and take n nearest
        distances.sort(key=lambda x: x[0])
        nearest = distances[:n_neighbors]

        # Inverse distance weighting with power=2
        total_weight = 0.0
        vx_weighted = 0.0
        vy_weighted = 0.0
        vz_weighted = 0.0

        for dist, station in nearest:
            # Avoid division by zero for exact matches
            if dist < 0.001:  # Within 1 meter
                return (station.vx, station.vy, station.vz)

            weight = 1.0 / (dist ** 2)
            total_weight += weight
            vx_weighted += weight * station.vx
            vy_weighted += weight * station.vy
            vz_weighted += weight * station.vz

        vx = vx_weighted / total_weight
        vy = vy_weighted / total_weight
        vz = vz_weighted / total_weight

        return (vx, vy, vz)

    def get_stations_in_region(self, lat_min: float, lat_max: float,
                               lon_min: float, lon_max: float) -> List[CORSStation]:
        """Get all stations within a rectangular region"""
        return [s for s in self.stations
                if lat_min <= s.lat <= lat_max and lon_min <= s.lon <= lon_max]

    def get_stations_by_state(self, state_code: str) -> List[CORSStation]:
        """Get all stations in a state"""
        return [s for s in self.stations if s.state.upper() == state_code.upper()]

    def diagnose_coverage(self, lat: float, lon: float, radius_km: float = 500):
        """Diagnose CORS station coverage at a location"""
        nearby = []
        for station in self.stations:
            dist = haversine_distance(lat, lon, station.lat, station.lon)
            if dist <= radius_km:
                nearby.append((dist, station))

        nearby.sort(key=lambda x: x[0])

        print(f"\nCORS Coverage at ({lat:.4f}°N, {lon:.4f}°E)")
        print(f"Radius: {radius_km} km")
        print("-" * 80)

        if not nearby:
            print("❌ No CORS stations within radius!")
            return

        print(f"Found {len(nearby)} stations:\n")
        print(f"{'Site':<8} {'Distance':>10} {'Lat':>10} {'Lon':>11} {'Vx':>8} {'Vy':>8} {'Vz':>8}")
        print("-" * 80)

        for dist, station in nearby[:10]:  # Show top 10
            print(f"{station.site:<8} {dist:10.1f} km {station.lat:10.4f} {station.lon:11.4f} "
                  f"{station.vx:8.2f} {station.vy:8.2f} {station.vz:8.2f}")

    def apply_velocities_to_las(self, input_file: str, output_file: str,
                                source_epoch: float, target_epoch: float,
                                progress_callback=None) -> dict:
        """
        Apply CORS velocities to LAS file using IDW interpolation

        Args:
            input_file: Input LAS file path (ITRF2020 @ source_epoch, geographic)
            output_file: Output LAS file path
            source_epoch: Source epoch (e.g., 2010.0)
            target_epoch: Target epoch (e.g., 2020.0)
            progress_callback: Optional callback for progress (pct between 0-100)

        Returns:
            dict with statistics
        """
        import laspy
        import logging

        logging.info(f"Applying CORS velocities via IDW: {source_epoch} → {target_epoch}")

        # Read input LAS
        las_in = laspy.read(input_file)
        point_count = len(las_in.x)

        time_years = target_epoch - source_epoch
        chunk_size = 100000

        logging.info(f"✓ Loaded {len(self.stations)} CORS stations")
        logging.info(f"Processing {point_count:,} points in chunks of {chunk_size:,}...")

        # Extract coordinates
        lons = las_in.x.copy()
        lats = las_in.y.copy()
        heights = las_in.z.copy()

        lats_new = np.zeros(point_count)
        lons_new = np.zeros(point_count)
        heights_new = np.zeros(point_count)

        # Process in chunks
        for i in range(0, point_count, chunk_size):
            end = min(i + chunk_size, point_count)
            progress_pct = (i / point_count) * 100

            if progress_callback:
                progress_callback(progress_pct)

            logging.info(f"Progress: {progress_pct:.1f}% ({i:,}/{point_count:,} points)")

            chunk_lats = lats[i:end]
            chunk_lons = lons[i:end]
            chunk_heights = heights[i:end]
            chunk_len = len(chunk_lats)

            # Convert to ECEF
            chunk_x = np.zeros(chunk_len)
            chunk_y = np.zeros(chunk_len)
            chunk_z = np.zeros(chunk_len)

            for j in range(chunk_len):
                from htdp_velocity import toxyz
                x, y, z = toxyz(chunk_lats[j], chunk_lons[j], chunk_heights[j])
                chunk_x[j] = x
                chunk_y[j] = y
                chunk_z[j] = z

            # Get CORS velocities for each point
            vx_array = np.zeros(chunk_len)
            vy_array = np.zeros(chunk_len)
            vz_array = np.zeros(chunk_len)

            for j in range(chunk_len):
                vel = self.get_velocity_xyz(chunk_lats[j], chunk_lons[j],
                                            n_neighbors=4, min_neighbors=1)
                if vel:
                    vx_array[j], vy_array[j], vz_array[j] = vel
                else:
                    # Fallback to zero if no stations nearby
                    vx_array[j], vy_array[j], vz_array[j] = 0.0, 0.0, 0.0

            # Apply displacement
            dx_array = (vx_array / 1000.0) * time_years  # Convert mm/yr to m/yr
            dy_array = (vy_array / 1000.0) * time_years
            dz_array = (vz_array / 1000.0) * time_years

            chunk_x_out = chunk_x + dx_array
            chunk_y_out = chunk_y + dy_array
            chunk_z_out = chunk_z + dz_array

            # Convert back to geodetic
            for j in range(chunk_len):
                lat_out, lon_out, h_out = xyz_to_geodetic(
                    chunk_x_out[j], chunk_y_out[j], chunk_z_out[j]
                )
                lats_new[i + j] = lat_out
                lons_new[i + j] = lon_out
                heights_new[i + j] = h_out

        # Write output
        las_out = laspy.read(input_file)
        las_out.x = lons_new
        las_out.y = lats_new
        las_out.z = heights_new
        las_out.write(output_file)

        logging.info(f"✓ CORS velocity interpolation complete")

        return {
            'velocity_source': 'CORS_IDW',
            'stations_used': len(self.stations),
            'points_processed': point_count
        }


def xyz_velocity_to_neu(vx: float, vy: float, vz: float,
                        lat_rad: float, lon_rad: float) -> Tuple[float, float, float]:
    """
    Convert XYZ velocity to NEU (North-East-Up) velocity

    Args:
        vx, vy, vz: Velocity in XYZ (mm/yr)
        lat_rad, lon_rad: Geodetic coordinates in radians

    Returns:
        (vN, vE, vU) in mm/yr
    """
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # Rotation matrix from XYZ to NEU
    vN = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz
    vE = -sin_lon * vx + cos_lon * vy
    vU = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz

    return (vN, vE, vU)


# Example usage
if __name__ == "__main__":
    # Example: Parse CORS data
    print("CORS Velocity Interpolator - Example Usage\n")
    print("=" * 80)

    # This would load your actual CORS data file
    # stations = parse_cors_velocity_file("cors_velocities.txt")
    # interpolator = CORSVelocityInterpolator(stations)

    # Example queries
    print("\nExample queries you can make:")
    print("-" * 80)
    print("1. Get velocity at a point:")
    print("   vx, vy, vz = interpolator.get_velocity_xyz(30.5, -91.1)")
    print("\n2. Check coverage:")
    print("   interpolator.diagnose_coverage(30.5, -91.1, radius_km=300)")
    print("\n3. Get stations in Louisiana:")
    print("   la_stations = interpolator.get_stations_by_state('LA')")
    print("\n4. Convert to NEU:")
    print("   vN, vE, vU = xyz_velocity_to_neu(vx, vy, vz, lat_rad, lon_rad)")