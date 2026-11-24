"""
CORS Data Integration for Velocity Calibration

This module loads NOAA CORS Multi-Year Solution 3 (MYCS3) data
and uses it to calibrate/validate velocity grids.
"""

import math
from collections import defaultdict


class CORSData:
    """Load and manage CORS station velocity data"""

    def __init__(self, cors_file_path):
        """
        Load CORS data from NOAA MYCS3 file

        Format:
        AZBH  2020.00  -2174940.110  -4749388.743   3647716.483   -15.5     2.3    -7.5    US      AZ      Operational

        Fields:
        1. Station ID
        2. Reference epoch
        3. X (m)
        4. Y (m)
        5. Z (m)
        6. Vx (mm/yr)
        7. Vy (mm/yr)
        8. Vz (mm/yr)
        9. Country
        10. State
        11. Status
        """
        self.stations = {}
        self.stations_by_state = defaultdict(list)

        with open(cors_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 11:
                    continue

                station_id = parts[0]
                epoch = float(parts[1])
                X = float(parts[2])
                Y = float(parts[3])
                Z = float(parts[4])
                Vx = float(parts[5])  # mm/yr
                Vy = float(parts[6])  # mm/yr
                Vz = float(parts[7])  # mm/yr
                country = parts[8]
                state = parts[9]
                status = parts[10] if len(parts) > 10 else "Unknown"

                # Convert XYZ to lat/lon/h for easier lookup
                lat, lon, h = self.xyz_to_geodetic(X, Y, Z)

                station_data = {
                    'id': station_id,
                    'epoch': epoch,
                    'X': X,
                    'Y': Y,
                    'Z': Z,
                    'Vx': Vx,
                    'Vy': Vy,
                    'Vz': Vz,
                    'lat': lat,
                    'lon': lon,
                    'h': h,
                    'country': country,
                    'state': state,
                    'status': status
                }

                self.stations[station_id] = station_data
                self.stations_by_state[state].append(station_data)

    @staticmethod
    def xyz_to_geodetic(X, Y, Z):
        """Convert ECEF XYZ to geodetic coordinates (WGS84)"""
        # Constants
        a = 6378137.0  # semi-major axis (meters)
        e2 = 0.00669438002290  # first eccentricity squared

        p = math.sqrt(X ** 2 + Y ** 2)
        lat_rad = math.atan2(Z, p * (1 - e2))

        # Iterate for latitude
        for _ in range(5):
            N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
            lat_rad = math.atan2(Z + e2 * N * math.sin(lat_rad), p)

        N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
        h = p / math.cos(lat_rad) - N

        lon_rad = math.atan2(Y, X)

        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)

        return lat, lon, h

    def get_stations_by_state(self, state_code):
        """Get all CORS stations in a given state"""
        return self.stations_by_state.get(state_code, [])

    def get_nearest_station(self, lat, lon, state_code=None, max_distance=1.0):
        """
        Find nearest CORS station to given location

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            state_code: Optional state filter (e.g., 'AZ', 'UT')
            max_distance: Maximum distance in degrees

        Returns:
            Station data dict or None
        """
        stations_to_search = []

        if state_code:
            stations_to_search = self.stations_by_state.get(state_code, [])
        else:
            stations_to_search = list(self.stations.values())

        nearest = None
        min_dist = float('inf')

        for station in stations_to_search:
            # Simple Euclidean distance in lat/lon (good enough for local searches)
            dist = math.sqrt((station['lat'] - lat) ** 2 + (station['lon'] - lon) ** 2)

            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = station

        return nearest, min_dist if nearest else (None, float('inf'))

    def get_nearest_stations(self, lat, lon, n=4, state_code=None, max_distance=1.5):
        """
        Find n nearest CORS stations to given location

        Returns:
            List of (station, distance) tuples, sorted by distance
        """
        stations_to_search = []

        if state_code:
            stations_to_search = self.stations_by_state.get(state_code, [])
        else:
            stations_to_search = list(self.stations.values())

        distances = []
        for station in stations_to_search:
            dist = math.sqrt((station['lat'] - lat) ** 2 + (station['lon'] - lon) ** 2)
            if dist <= max_distance:
                distances.append((station, dist))

        # Sort by distance and return top n
        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def interpolate_velocity(self, lat, lon, n=4, state_code=None):
        """
        Interpolate velocity at given location using nearest CORS stations

        Uses inverse distance weighting (IDW) for interpolation

        Args:
            lat: Latitude
            lon: Longitude
            n: Number of stations to use
            state_code: Optional state filter

        Returns:
            (Vx, Vy, Vz) in mm/yr, or None if no stations found
        """
        nearest = self.get_nearest_stations(lat, lon, n=n, state_code=state_code)

        if not nearest:
            return None

        # Inverse distance weighting
        total_weight = 0
        weighted_vx = 0
        weighted_vy = 0
        weighted_vz = 0

        for station, dist in nearest:
            if dist < 0.0001:  # At a station - use exactly
                return (station['Vx'], station['Vy'], station['Vz'])

            # Weight inversely proportional to distance squared
            weight = 1.0 / (dist ** 2)
            total_weight += weight
            weighted_vx += station['Vx'] * weight
            weighted_vy += station['Vy'] * weight
            weighted_vz += station['Vz'] * weight

        if total_weight == 0:
            return None

        return (weighted_vx / total_weight, weighted_vy / total_weight, weighted_vz / total_weight)

    def print_state_summary(self, state_code):
        """Print summary of CORS stations in a state"""
        stations = self.stations_by_state.get(state_code, [])
        print(f"\n{'=' * 80}")
        print(f"CORS Stations in {state_code}: {len(stations)} stations")
        print(f"{'=' * 80}")
        print(f"{'ID':<8} {'Lat':<10} {'Lon':<12} {'Vx':<8} {'Vy':<8} {'Vz':<8}")
        print(f"{'-' * 80}")

        for station in sorted(stations, key=lambda s: s['lat'], reverse=True):
            print(f"{station['id']:<8} {station['lat']:>9.4f} {station['lon']:>11.4f} "
                  f"{station['Vx']:>7.1f} {station['Vy']:>7.1f} {station['Vz']:>7.1f}")

        # Statistics
        if stations:
            vx_vals = [s['Vx'] for s in stations]
            vy_vals = [s['Vy'] for s in stations]
            vz_vals = [s['Vz'] for s in stations]

            print(f"{'-' * 80}")
            print(f"{'Mean':<8} {'':<10} {'':<12} {sum(vx_vals) / len(vx_vals):>7.1f} "
                  f"{sum(vy_vals) / len(vy_vals):>7.1f} {sum(vz_vals) / len(vz_vals):>7.1f}")


def main():
    """Test CORS data loading"""
    import sys

    cors_file = r"D:\Winter2025\SPCS2022\NATRF22_GUI\corsdata_itrf2020_xyzcomp.txt"

    print("Loading CORS data...")
    cors = CORSData(cors_file)

    print(f"Loaded {len(cors.stations)} total CORS stations")

    # Print summaries for key states
    for state in ['AZ', 'UT', 'AR', 'MS', 'TX', 'LA']:
        cors.print_state_summary(state)

    # Example: Interpolate velocity at a test location
    print(f"\n{'=' * 80}")
    print("Example: Interpolate velocity at Salt Lake City area")
    print(f"{'=' * 80}")

    slc_lat, slc_lon = 42.23, -110.03  # Salt Lake City

    # Find nearest stations
    nearest = cors.get_nearest_stations(slc_lat, slc_lon, n=4, state_code='UT')
    print(f"\nNearest 4 stations to SLC ({slc_lat}, {slc_lon}):")
    for station, dist in nearest:
        print(f"  {station['id']:8s}: {dist:6.3f}° away, "
              f"Vx={station['Vx']:7.1f}, Vy={station['Vy']:7.1f}, Vz={station['Vz']:7.1f} mm/yr")

    # Interpolate
    vx, vy, vz = cors.interpolate_velocity(slc_lat, slc_lon, n=4, state_code='UT')
    print(f"\nInterpolated velocity at SLC:")
    print(f"  Vx = {vx:7.1f} mm/yr")
    print(f"  Vy = {vy:7.1f} mm/yr")
    print(f"  Vz = {vz:7.1f} mm/yr")


if __name__ == '__main__':
    main()