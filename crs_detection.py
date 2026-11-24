"""
Comprehensive State Plane CRS Detection using Geographic State Bounds
Integrates with state_bounds.py for accurate automatic detection
"""

import numpy as np
from pyproj import CRS, Transformer
import logging
from state_bounds import STATE_BOUNDS, get_state_from_coords

logger = logging.getLogger(__name__)


class StatePlaneCRSDetector:
    """
    Automatic State Plane CRS detection using coordinate analysis
    """

    # NAD83(2011) State Plane EPSG Codes with coordinate ranges
    # Format: EPSG: (state, zone_name, E_min, E_max, N_min, N_max)
    # Coordinate ranges in METERS for typical data coverage

    STATE_PLANE_ZONES = {
        # ALASKA - NAD83 - 10 zones
        26931: ('AK', 'Alaska Zone 1', 400000, 900000, 1400000, 2100000),
        26932: ('AK', 'Alaska Zone 2', 400000, 900000, 1000000, 1800000),
        26933: ('AK', 'Alaska Zone 3', 400000, 900000, 900000, 1600000),
        26934: ('AK', 'Alaska Zone 4', 400000, 900000, 800000, 1500000),
        26935: ('AK', 'Alaska Zone 5', 200000, 900000, 800000, 1800000),
        26936: ('AK', 'Alaska Zone 6', 400000, 900000, 800000, 1500000),
        26937: ('AK', 'Alaska Zone 7', 400000, 900000, 800000, 1600000),
        26938: ('AK', 'Alaska Zone 8', 200000, 900000, 700000, 1700000),
        26939: ('AK', 'Alaska Zone 9', 400000, 900000, 700000, 1500000),
        26940: ('AK', 'Alaska Zone 10', 900000, 1500000, 1300000, 2000000),

        # CALIFORNIA - NAD83(2011) - 6 zones
        6415: ('CA', 'California Zone 1', 1800000, 2600000, 400000, 800000),
        6417: ('CA', 'California Zone 2', 1800000, 2600000, 300000, 700000),
        6419: ('CA', 'California Zone 3', 1800000, 2600000, 100000, 700000),
        6421: ('CA', 'California Zone 4', 1800000, 2600000, 100000, 600000),
        6423: ('CA', 'California Zone 5', 1800000, 2600000, 100000, 600000),
        6425: ('CA', 'California Zone 6', 1800000, 2600000, 100000, 500000),

        # OREGON - NAD83(2011) - 2 zones
        6558: ('OR', 'Oregon North', 100000, 900000, 100000, 500000),
        6560: ('OR', 'Oregon South', 100000, 900000, 100000, 500000),

        # WASHINGTON - NAD83(2011) - 2 zones
        6596: ('WA', 'Washington North', 500000, 3000000, 0, 400000),
        6598: ('WA', 'Washington South', 500000, 3000000, 0, 400000),

        # ARIZONA - NAD83(2011) - 3 zones
        6406: ('AZ', 'Arizona East', 100000, 900000, 0, 1000000),
        6404: ('AZ', 'Arizona Central', 100000, 900000, 0, 1000000),
        6408: ('AZ', 'Arizona West', 100000, 900000, 0, 1000000),

        # COLORADO - NAD83(2011) - 3 zones
        6429: ('CO', 'Colorado North', 100000, 900000, 1000000, 2000000),
        6427: ('CO', 'Colorado Central', 100000, 900000, 1000000, 2000000),
        6431: ('CO', 'Colorado South', 100000, 900000, 1000000, 2000000),

        # IDAHO - NAD83(2011) - 3 zones
        6450: ('ID', 'Idaho East', 100000, 900000, 0, 1000000),
        6448: ('ID', 'Idaho Central', 100000, 900000, 0, 1000000),
        6452: ('ID', 'Idaho West', 100000, 900000, 0, 1000000),

        # MONTANA - NAD83(2011) - 1 zone
        6514: ('MT', 'Montana', 100000, 900000, 0, 1500000),

        # NEVADA - NAD83(2011) - 3 zones
        6520: ('NV', 'Nevada East', 100000, 900000, 1400000, 2500000),
        6518: ('NV', 'Nevada Central', 100000, 900000, 1400000, 2500000),
        6522: ('NV', 'Nevada West', 100000, 900000, 1400000, 2500000),

        # NEW MEXICO - NAD83(2011) - 3 zones
        6530: ('NM', 'New Mexico East', 100000, 900000, 0, 1500000),
        6528: ('NM', 'New Mexico Central', 100000, 900000, 0, 1500000),
        6532: ('NM', 'New Mexico West', 100000, 900000, 0, 1500000),

        # TEXAS - NAD83(2011) - 5 zones
        6581: ('TX', 'Texas North', 100000, 1000000, 7000000, 10000000),
        6583: ('TX', 'Texas North Central', 600000, 3000000, 6000000, 8000000),
        6577: ('TX', 'Texas Central', 600000, 3000000, 3000000, 5000000),
        6587: ('TX', 'Texas South Central', 600000, 3000000, 2000000, 4000000),
        6585: ('TX', 'Texas South', 100000, 1000000, 2000000, 7000000),

        # UTAH - NAD83(2011) - 3 zones
        6620: ('UT', 'Utah North', 100000, 900000, 1000000, 2000000),
        6621: ('UT', 'Utah Central', 100000, 900000, 1000000, 2000000),
        6619: ('UT', 'Utah South', 100000, 900000, 1000000, 2000000),

        # WYOMING - NAD83(2011) - 4 zones
        6611: ('WY', 'Wyoming East', 100000, 900000, 0, 1000000),
        6613: ('WY', 'Wyoming East Central', 100000, 900000, 0, 1000000),
        6617: ('WY', 'Wyoming West Central', 100000, 900000, 0, 1000000),
        6616: ('WY', 'Wyoming West', 100000, 900000, 0, 1000000),

        # Add more states as needed...
        # Complete list available at: https://epsg.org/
    }

    @classmethod
    def detect_state_from_geographic(cls, lat, lon):
        """
        Detect state from geographic coordinates

        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees, negative for west)

        Returns:
            2-letter state code or None
        """
        state = get_state_from_coords(lat, lon)
        if state:
            logger.info(f"  Geographic location: {state}")
        return state

    @classmethod
    def detect_state_from_projected(cls, x_min, x_max, y_min, y_max):
        """
        Detect possible states from projected coordinate ranges

        Args:
            x_min, x_max: Easting range
            y_min, y_max: Northing range

        Returns:
            list of possible state codes
        """
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Get all zones that could contain these coordinates
        candidates = []
        for epsg, (state, zone, e_min, e_max, n_min, n_max) in cls.STATE_PLANE_ZONES.items():
            # Check if data center falls within zone
            if e_min <= x_center <= e_max and n_min <= y_center <= n_max:
                candidates.append(state)

        # Return unique states
        return list(set(candidates))

    @classmethod
    def detect_zone_from_coordinates(cls, x_coords, y_coords, state_filter=None):
        """
        Detect State Plane zone from coordinate ranges

        Args:
            x_coords: Array of X/Easting coordinates
            y_coords: Array of Y/Northing coordinates
            state_filter: Optional state code to filter results

        Returns:
            list of (epsg, state, zone_name, confidence) tuples
        """
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_range = x_max - x_min
        y_range = y_max - y_min

        logger.info(f"Coordinate ranges:")
        logger.info(f"  X: {x_min:,.1f} to {x_max:,.1f} (range: {x_range:,.1f} m)")
        logger.info(f"  Y: {y_min:,.1f} to {y_max:,.1f} (range: {y_range:,.1f} m)")
        logger.info(f"  Center: ({x_center:,.1f}, {y_center:,.1f})")
        logger.info("")

        candidates = []

        for epsg, (state, zone_name, e_min, e_max, n_min, n_max) in cls.STATE_PLANE_ZONES.items():
            # Skip if state filter doesn't match
            if state_filter and state != state_filter:
                continue

            # Check if data center point falls within zone bounds
            x_in_range = e_min <= x_center <= e_max
            y_in_range = n_min <= y_center <= n_max

            if x_in_range and y_in_range:
                # Calculate confidence based on margins
                x_margin = min(x_center - e_min, e_max - x_center)
                y_margin = min(y_center - n_min, n_max - y_center)

                # Normalize by expected range
                x_confidence = x_margin / (e_max - e_min) if (e_max - e_min) > 0 else 0
                y_confidence = y_margin / (n_max - n_min) if (n_max - n_min) > 0 else 0

                # Overall confidence is minimum of the two
                confidence = min(x_confidence, y_confidence)

                # Bonus: penalize if data range is much larger than zone
                if x_range > 0.8 * (e_max - e_min) or y_range > 0.8 * (n_max - n_min):
                    confidence *= 0.8  # Reduce confidence

                candidates.append((epsg, state, zone_name, confidence))

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x[3], reverse=True)

        return candidates

    @classmethod
    def detect_crs(cls, las, user_hint_epsg=None):
        """
        Main CRS detection with geographic state lookup

        Args:
            las: laspy LAS object
            user_hint_epsg: Optional user-provided EPSG code

        Returns:
            CRS object
        """
        logger.info("=" * 80)
        logger.info("AUTOMATIC CRS DETECTION")
        logger.info("=" * 80)

        # Step 1: Check header CRS
        try:
            header_crs = las.header.parse_crs()
            if header_crs:
                logger.info(f"Header CRS: {header_crs.name}")
                logger.info(f"  Type: {header_crs.type_name}")
                logger.info(f"  Is Geographic: {header_crs.is_geographic}")
                logger.info(f"  Is Projected: {header_crs.is_projected}")
            else:
                logger.info("Header CRS: None")
                header_crs = None
        except Exception as e:
            logger.warning(f"Error reading header CRS: {e}")
            header_crs = None

        logger.info("")

        # Step 2: Analyze coordinate ranges
        x_min, x_max = np.min(las.x), np.max(las.x)
        y_min, y_max = np.min(las.y), np.max(las.y)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        coords_are_geographic = (-180 <= x_min <= 180) and (-90 <= y_min <= 90)
        coords_are_projected = (
                (10000 < abs(x_min) < 10000000) and
                (10000 < abs(y_min) < 10000000)
        )

        logger.info("Coordinate Type Analysis:")
        if coords_are_geographic:
            logger.info("  ✓ Coordinates appear GEOGRAPHIC (lat/lon)")
            logger.info(f"    Longitude: {x_min:.6f} to {x_max:.6f}")
            logger.info(f"    Latitude:  {y_min:.6f} to {y_max:.6f}")
        elif coords_are_projected:
            logger.info("  ✓ Coordinates appear PROJECTED (State Plane)")
            logger.info(f"    Easting:  {x_min:,.1f} to {x_max:,.1f} m")
            logger.info(f"    Northing: {y_min:,.1f} to {y_max:,.1f} m")
        else:
            logger.error("  ✗ Coordinates are INVALID/CORRUPT")
            logger.error(f"    X: {x_min} to {x_max}")
            logger.error(f"    Y: {y_min} to {y_max}")
            raise ValueError("Cannot process invalid coordinates")

        logger.info("")

        # Step 3: Check for CRS mismatch
        if header_crs and header_crs.is_geographic and coords_are_projected:
            logger.warning("⚠ CRS MISMATCH DETECTED!")
            logger.warning("  Header says: Geographic")
            logger.warning("  Coordinates are: Projected")
            logger.warning("  → Ignoring corrupt header metadata")
            header_crs = None

        # Step 4: User hint validation
        if user_hint_epsg:
            try:
                hint_crs = CRS.from_epsg(user_hint_epsg)
                logger.info(f"User provided EPSG:{user_hint_epsg} ({hint_crs.name})")

                # Validate hint type matches coordinates
                if coords_are_geographic and hint_crs.is_projected:
                    logger.warning("  ⚠ Hint is projected but coords are geographic - IGNORING")
                elif coords_are_projected and hint_crs.is_geographic:
                    logger.warning("  ⚠ Hint is geographic but coords are projected - IGNORING")
                else:
                    logger.info("  ✓ User hint validated")
                    logger.info("=" * 80)
                    return hint_crs
            except Exception as e:
                logger.warning(f"Invalid EPSG code {user_hint_epsg}: {e}")

        # Step 5: Try to use header if it matches coordinate type
        if header_crs:
            if (header_crs.is_geographic and coords_are_geographic) or \
                    (header_crs.is_projected and coords_are_projected):
                logger.info(f"✓ Using header CRS: {header_crs.name}")
                logger.info("=" * 80)
                return header_crs

        # Step 6: Automatic detection
        logger.info("Automatic Zone Detection:")
        logger.info("-" * 40)

        detected_state = None

        # For geographic coordinates, detect state directly
        if coords_are_geographic:
            detected_state = cls.detect_state_from_geographic(y_center, x_center)
            if detected_state:
                logger.info(f"  State: {detected_state}")
                logger.info("  → Using NAD83(2011) geographic (EPSG:6319)")
                logger.info("=" * 80)
                return CRS.from_epsg(6319)
            else:
                logger.warning("  Could not determine state from coordinates")
                logger.info("  → Defaulting to NAD83(2011) geographic (EPSG:6319)")
                logger.info("=" * 80)
                return CRS.from_epsg(6319)

        # For projected coordinates, detect zone
        if coords_are_projected:
            # First try to narrow down by state using projected coords
            possible_states = cls.detect_state_from_projected(x_min, x_max, y_min, y_max)

            if len(possible_states) == 1:
                detected_state = possible_states[0]
                logger.info(f"  Narrowed to state: {detected_state}")
            elif len(possible_states) > 1:
                logger.info(f"  Multiple possible states: {', '.join(possible_states)}")
                detected_state = possible_states[0]  # Use first match

            # Detect specific zone
            candidates = cls.detect_zone_from_coordinates(
                las.x, las.y,
                state_filter=detected_state
            )

            if candidates:
                logger.info(f"Found {len(candidates)} candidate zone(s):")
                for i, (epsg, state, zone, conf) in enumerate(candidates[:5]):
                    logger.info(f"  {i + 1}. EPSG:{epsg} - {state} {zone} (confidence: {conf:.1%})")

                # Use best match
                best_epsg, best_state, best_zone, best_conf = candidates[0]

                logger.info("")
                if best_conf > 0.3:
                    logger.info(f"✓ Auto-detected: EPSG:{best_epsg}")
                    logger.info(f"  Zone: {best_state} {best_zone}")
                    logger.info(f"  Confidence: {best_conf:.1%}")
                else:
                    logger.warning(f"⚠ Low confidence ({best_conf:.1%})")
                    logger.info(f"  Using best guess: EPSG:{best_epsg} - {best_state} {best_zone}")

                logger.info("=" * 80)
                return CRS.from_epsg(best_epsg)
            else:
                raise ValueError(
                    f"Could not detect State Plane zone!\n"
                    f"Coordinates: X={x_min:,.1f} to {x_max:,.1f}\n"
                    f"             Y={y_min:,.1f} to {y_max:,.1f}\n"
                    f"Possible states: {possible_states}\n"
                    f"Please specify EPSG code manually."
                )

        raise ValueError("CRS detection failed - please specify EPSG code manually")

