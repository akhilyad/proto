from typing import Any, Tuple, List, Dict, Optional, Callable
import folium
from folium.plugins import MarkerCluster, AntPath, Fullscreen, MeasureControl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from folium import Element, DivIcon
import requests
import json
import os
from geopy.distance import geodesic
import math
import openrouteservice
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd
from datetime import datetime, timedelta
import pytz
from dataclasses import dataclass
from heapq import heappush, heappop
from collections import defaultdict
import networkx as nx
import random
from requests.exceptions import RequestException, Timeout
import logging
from dotenv import load_dotenv
import time
from functools import lru_cache
import streamlit as st
from openrouteservice import client
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import heapq
import pickle
from rapidfuzz import process, fuzz

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure requests session with retries and timeouts
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Get API key from environment variable
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
if not WEATHER_API_KEY:
    logging.warning("Weather API key not found in environment variables. Weather features will be disabled.")

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    pass

class WeatherAPIError(APIError):
    """Exception raised for weather API specific errors."""
    pass

# Initialize OpenRouteService client
ors_client = openrouteservice.Client(key=os.getenv('ORS_API_KEY'))

# Constants for route planning
CURRENT_TIME = datetime.now(pytz.timezone('Europe/Paris'))  # 10:59 PM CEST, May 21, 2025
MONSOON_SEASON = CURRENT_TIME.month in [6, 7, 8, 9]  # Monsoon season in Indian Ocean
ROARING_FORTIES = (-50, -40)  # Latitude range for Roaring Forties
PIRACY_ZONES = [
    Polygon([(10, 45), (15, 45), (15, 55), (10, 55)]),  # Gulf of Aden
    Polygon([(0, 45), (5, 45), (5, 55), (0, 55)])       # Somali Basin
]

# Major ports for accurate marker placement and routing
MAJOR_PORTS = {
    "Hamburg": (53.5511, 9.9937),
    "Port Said": (31.2653, 32.3019),
    "Mumbai": (19.0760, 72.8777),
    # Add more as needed
}

# Expanded global waypoint network for realistic maritime routing
WAYPOINTS = {
    "Hamburg": {"coords": (53.5511, 9.9937), "neighbors": ["Rotterdam", "Dover"]},
    "Rotterdam": {"coords": (51.9244, 4.4777), "neighbors": ["Hamburg", "Gibraltar", "New York"]},
    "Dover": {"coords": (50.9631, 1.8526), "neighbors": ["Hamburg", "Gibraltar"]},
    "Gibraltar": {"coords": (36.1408, -5.3536), "neighbors": ["Dover", "Port Said", "Cape Town"]},
    "Port Said": {"coords": (31.2653, 32.3019), "neighbors": ["Gibraltar", "Aden", "Dubai"]},
    "Aden": {"coords": (12.7855, 45.0187), "neighbors": ["Port Said", "Mumbai", "Singapore"]},
    "Mumbai": {"coords": (19.0760, 72.8777), "neighbors": ["Aden", "Singapore"]},
    "Singapore": {"coords": (1.3521, 103.8198), "neighbors": ["Aden", "Shanghai", "Hong Kong"]},
    "Shanghai": {"coords": (31.2304, 121.4737), "neighbors": ["Singapore", "Hong Kong"]},
    "Hong Kong": {"coords": (22.3193, 114.1694), "neighbors": ["Singapore", "Shanghai", "Sydney"]},
    "Sydney": {"coords": (-33.8688, 151.2093), "neighbors": ["Hong Kong", "Santos"]},
    "Santos": {"coords": (-23.9608, -46.3336), "neighbors": ["Sydney", "Cape Town", "New York"]},
    "Cape Town": {"coords": (-33.9249, 18.4241), "neighbors": ["Gibraltar", "Santos", "Dubai"]},
    "Dubai": {"coords": (25.2048, 55.2708), "neighbors": ["Port Said", "Cape Town", "Singapore"]},
    "New York": {"coords": (40.7128, -74.0060), "neighbors": ["Rotterdam", "Santos", "Los Angeles"]},
    "Los Angeles": {"coords": (33.7406, -118.2770), "neighbors": ["New York", "Shanghai"]},
    # Add more ports and connections for greater realism
}

# --- Caching/loading for large datasets ---
def build_waypoints_from_csv(csv_path, cache_path='waypoints_cache.pkl'):
    try:
        # Try to load from cache
        with open(cache_path, 'rb') as f:
            waypoints = pickle.load(f)
        return waypoints
    except Exception:
        pass
    # If cache not found, build from CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    waypoints = {}
    for _, row in df.iterrows():
        src = row['source_port']
        dst = row['destination_port']
        src_coords = (row['source_lat'], row['source_lon'])
        dst_coords = (row['destination_lat'], row['destination_lon'])
        if src not in waypoints:
            waypoints[src] = {'coords': src_coords, 'neighbors': set()}
        if dst not in waypoints:
            waypoints[dst] = {'coords': dst_coords, 'neighbors': set()}
        waypoints[src]['neighbors'].add(dst)
        waypoints[dst]['neighbors'].add(src)
    for port in waypoints:
        waypoints[port]['neighbors'] = list(waypoints[port]['neighbors'])
    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(waypoints, f)
    return waypoints

# Example usage:
# WAYPOINTS = build_waypoints_from_csv('path/to/unctad_port_to_port.csv')

def fuzzy_find_port_name(query, waypoints):
    # Use rapidfuzz to find the closest port name in the network
    port_names = list(waypoints.keys())
    match, score, _ = process.extractOne(query, port_names, scorer=fuzz.WRatio)
    return match if score > 70 else None  # threshold can be adjusted

# Update find_nearest_waypoint to use fuzzy matching for user-supplied names
# (for lat/lon input, keep as before)
def find_nearest_waypoint(lat, lon, waypoints=WAYPOINTS):
    min_dist = float('inf')
    nearest = None
    for name, data in waypoints.items():
        d = geodesic((lat, lon), data["coords"]).km
        if d < min_dist:
            min_dist = d
            nearest = name
    return nearest

def find_route(start_name, end_name):
    queue = [(0, start_name, [start_name])]
    visited = set()
    while queue:
        cost, current, path = heapq.heappop(queue)
        if current == end_name:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor in WAYPOINTS[current]["neighbors"]:
            if neighbor not in visited:
                n_coords = WAYPOINTS[neighbor]["coords"]
                c_coords = WAYPOINTS[current]["coords"]
                n_cost = cost + geodesic(c_coords, n_coords).km
                heapq.heappush(queue, (n_cost, neighbor, path + [neighbor]))
    return []

@dataclass
class Location:
    """Represents a location with its type and coordinates."""
    name: str
    country: str
    coordinates: Tuple[float, float]
    type: str  # 'inland', 'coastal', 'port'
    port_name: Optional[str] = None
    rail_hub: Optional[str] = None

@dataclass
class RouteSegment:
    """Represents a segment of the route with its mode and details."""
    mode: str
    start: Location
    end: Location
    path: List[Tuple[float, float]]
    distance: float  # in kilometers
    duration: float  # in hours
    co2_emissions: float  # in kg
    constraints: Dict[str, Any]
    switchover_time: float = 0  # in hours

@dataclass
class RouteLeg:
    """Represents a complete leg of the journey."""
    segments: List[RouteSegment]
    total_distance: float
    total_duration: float
    total_co2: float
    switchover_points: List[Location]

class MultiModalRouter:
    """Handles multi-modal route planning with real-world constraints."""
    
    def __init__(self):
        self.ors_client = openrouteservice.Client(key=os.getenv('ORS_API_KEY'))
        self.weather_api_key = WEATHER_API_KEY
        self.marine_api_key = os.getenv('MARINETRAFFIC_API_KEY')
        self.request_timeout = 5  # seconds
        self.max_retries = 3
        self.retry_delay = 1  # second
        self._last_request_time = 0
        self._min_request_interval = 0.1  # seconds between requests
        
        # Mode-specific parameters
        self.mode_params = {
            'Truck': {
                'avg_speed': 60,  # km/h
                'max_range': 400,  # km for electric trucks
                'charging_time': 0.75,  # hours
                'co2_per_km': 0.8  # kg/km
            },
            'Train': {
                'avg_speed': 100,  # km/h
                'loading_time': 12,  # hours
                'co2_per_km': 0.3  # kg/km
            },
            'Ship': {
                'avg_speed': 27.8,  # km/h (15 knots)
                'port_time': 24,  # hours
                'co2_per_km': 0.02  # kg/km
            }
        }
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last_request)
        self._last_request_time = time.time()

    @lru_cache(maxsize=100)
    def get_weather_conditions(self, coordinates: Tuple[float, float]) -> Dict[str, Any]:
        """
        Get current weather conditions for a location with caching and rate limiting.
        
        Args:
            coordinates: Tuple of (latitude, longitude)
            
        Returns:
            Dict containing weather conditions or empty dict if unavailable
            
        Raises:
            WeatherAPIError: If there's an error with the weather API
            RateLimitError: If rate limit is exceeded
        """
        if not self.weather_api_key:
            logging.warning("Weather API key not configured")
            return {}
            
        self._rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': coordinates[0],
                    'lon': coordinates[1],
                    'appid': self.weather_api_key,
                    'units': 'metric'
                }
                
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 429:
                    raise RateLimitError("Weather API rate limit exceeded")
                response.raise_for_status()
                
                data = response.json()
                return {
                    'wind_speed': data['wind']['speed'],
                    'wave_height': data.get('waves', {}).get('height', 0),
                    'visibility': data['visibility'],
                    'weather': data['weather'][0]['main'],
                    'temperature': data['main']['temp']
                }
                
            except Timeout:
                if attempt == self.max_retries - 1:
                    logging.error("Weather API request timed out after all retries")
                    raise WeatherAPIError("Weather API request timed out")
                time.sleep(self.retry_delay * (attempt + 1))
                
            except RateLimitError:
                logging.error("Weather API rate limit exceeded")
                raise
                
            except RequestException as e:
                logging.error(f"Weather API request failed: {e}")
                raise WeatherAPIError(f"Weather API request failed: {str(e)}")
                
            except (KeyError, ValueError) as e:
                logging.error(f"Error parsing weather API response: {e}")
                raise WeatherAPIError(f"Invalid weather API response: {str(e)}")
                
        return {}
    
    def find_nearest_port(self, location: Location) -> Location:
        """Find the nearest port to a location."""
        # This would typically query a port database
        # For now, return a mock port
        return Location(
            name=f"Port of {location.name}",
            country=location.country,
            coordinates=(location.coordinates[0] + 0.1, location.coordinates[1] + 0.1),
            type='port',
            port_name=f"Port of {location.name}"
        )
    
    def find_nearest_rail_hub(self, location: Location) -> Location:
        """Find the nearest rail hub to a location."""
        # This would typically query a rail network database
        # For now, return a mock rail hub
        return Location(
            name=f"{location.name} Rail Hub",
            country=location.country,
            coordinates=(location.coordinates[0] + 0.05, location.coordinates[1] + 0.05),
            type='inland',
            rail_hub=f"{location.name} Rail Hub"
        )
    
    def check_maritime_hazards(self, coordinates: Tuple[float, float]) -> List[str]:
        """Check for maritime hazards at given coordinates."""
        hazards = []
        point = Point(coordinates[1], coordinates[0])
        
        # Check piracy zones
        for zone in PIRACY_ZONES:
            if zone.contains(point):
                hazards.append('piracy_risk')
        
        # Check Roaring Forties
        if ROARING_FORTIES[0] <= coordinates[0] <= ROARING_FORTIES[1]:
            hazards.append('rough_seas')
        
        # Check monsoon season in Indian Ocean
        if MONSOON_SEASON and 0 <= coordinates[0] <= 30 and 60 <= coordinates[1] <= 100:
            hazards.append('monsoon')
        
        return hazards
    
    def get_road_route(self, start: Location, end: Location, mode: str = 'Truck') -> RouteSegment:
        """Get road route using ORS API with vehicle constraints."""
        try:
            coords = [[start.coordinates[1], start.coordinates[0]], 
                     [end.coordinates[1], end.coordinates[0]]]
            
            params = {
                'preference': 'fastest',
                'geometry_simplify': False,
                'continue_straight': False,
                'optimize_waypoints': True,
                'avoid_features': ['highways', 'tollways'],
                'vehicle_type': 'hgv',
                'weight': 3.5,
                'height': 4.0,
                'width': 2.5,
                'length': 12.0,
                'axleload': 11.5,
                'elevation': True,
                'instructions': True
            }
            
            route = self.ors_client.directions(
                coordinates=coords,
                profile='driving-hgv',
                format='geojson',
                **params
            )
            
            if route and 'features' in route and len(route['features']) > 0:
                feature = route['features'][0]
                path = [(coord[1], coord[0]) for coord in feature['geometry']['coordinates']]
                distance = feature['properties']['segments'][0]['distance'] / 1000  # Convert to km
                duration = feature['properties']['segments'][0]['duration'] / 3600  # Convert to hours
                
                # Add charging stops for electric trucks
                if mode == 'Electric Truck':
                    charging_stops = self._add_charging_stops(path, distance)
                    path.extend(charging_stops)
                    duration += len(charging_stops) * self.mode_params['Truck']['charging_time']
                
                return RouteSegment(
                    mode=mode,
                    start=start,
                    end=end,
                    path=path,
                    distance=distance,
                    duration=duration,
                    co2_emissions=distance * self.mode_params[mode]['co2_per_km'],
                    constraints={'vehicle_type': 'electric' if mode == 'Electric Truck' else 'diesel'}
                )
            
        except Exception as e:
            print(f"Error calculating road route: {e}")
        
        return None
    
    def get_maritime_route(self, start: Location, end: Location) -> RouteSegment:
        """Get maritime route considering weather and hazards."""
        try:
            # Calculate great circle path
            path = self._create_maritime_path(start.coordinates, end.coordinates)
            
            # Adjust path for hazards
            adjusted_path = []
            for point in path:
                hazards = self.check_maritime_hazards(point)
                if hazards:
                    # Adjust route to avoid hazards
                    adjusted_point = self._adjust_for_hazards(point, hazards)
                    adjusted_path.append(adjusted_point)
                else:
                    adjusted_path.append(point)
            
            # Get weather conditions
            weather = self.get_weather_conditions(start.coordinates)
            
            # Calculate distance and duration
            distance = sum(geodesic(adjusted_path[i], adjusted_path[i+1]).kilometers 
                         for i in range(len(adjusted_path)-1))
            duration = distance / self.mode_params['Ship']['avg_speed']
            
            # Adjust duration for weather
            if weather.get('wave_height', 0) > 3:
                duration *= 1.2  # 20% slower in rough seas
            
            return RouteSegment(
                mode='Ship',
                start=start,
                end=end,
                path=adjusted_path,
                distance=distance,
                duration=duration,
                co2_emissions=distance * self.mode_params['Ship']['co2_per_km'],
                constraints={
                    'weather': weather,
                    'hazards': self.check_maritime_hazards(start.coordinates)
                }
            )
            
        except Exception as e:
            print(f"Error calculating maritime route: {e}")
        
        return None
    
    def _create_maritime_path(self, start: Tuple[float, float], end: Tuple[float, float], 
                            num_points: int = 100) -> List[Tuple[float, float]]:
        """Create a globally realistic maritime path using dynamic waypoints and pathfinding."""
        try:
            start_wp = find_nearest_waypoint(*start)
            end_wp = find_nearest_waypoint(*end)
            route_names = find_route(start_wp, end_wp)
            waypoints = [start] + [WAYPOINTS[name]["coords"] for name in route_names] + [end]
            path = []
            for i in range(len(waypoints) - 1):
                seg_start = waypoints[i]
                seg_end = waypoints[i+1]
                for j in range(num_points):
                    ratio = j / num_points
                    curve = 2 * math.sin(ratio * math.pi)
                    lat = seg_start[0] + (seg_end[0] - seg_start[0]) * ratio + curve
                    lon = seg_start[1] + (seg_end[1] - seg_start[1]) * ratio
                    path.append((lat, lon))
            path.append(end)
            return path
        except Exception as e:
            logger.error(f"Error creating maritime path: {e}")
            return [start, end]  # Fallback to direct path
    
    def _adjust_for_hazards(self, point: Tuple[float, float], hazards: List[str]) -> Tuple[float, float]:
        """Adjust route point to avoid hazards."""
        lat, lon = point
        if 'piracy_risk' in hazards:
            # Move north to avoid piracy zones
            lat += 0.5
        if 'rough_seas' in hazards:
            # Move north to avoid Roaring Forties
            lat += 1.0
        if 'monsoon' in hazards:
            # Adjust route based on monsoon season
            lat += 0.3
        return (lat, lon)
    
    def _add_charging_stops(self, path: List[Tuple[float, float]], total_distance: float) -> List[Tuple[float, float]]:
        """Add charging stops for electric trucks."""
        stops = []
        max_range = self.mode_params['Truck']['max_range']
        num_stops = math.ceil(total_distance / max_range)
        
        for i in range(1, num_stops):
            stop_index = int(len(path) * (i / num_stops))
            stops.append(path[stop_index])
        
        return stops
    
    def plan_route(self, source: Location, destination: Location, 
                  intermediate_stops: List[Location] = None) -> RouteLeg:
        """Plan a complete multi-modal route, including intercontinental (transoceanic) cases."""
        if intermediate_stops is None:
            intermediate_stops = []

        # Helper: very rough continent check by longitude/latitude
        def get_continent(lat, lon):
            if -10 <= lon <= 60 and 35 <= lat <= 70:
                return 'Europe'
            if -90 <= lon <= -30 and -60 <= lat <= 15:
                return 'SouthAmerica'
            if -130 <= lon <= -60 and 15 <= lat <= 60:
                return 'NorthAmerica'
            if 60 <= lon <= 150 and -10 <= lat <= 60:
                return 'Asia'
            if 110 <= lon <= 155 and -45 <= lat <= -10:
                return 'Australia'
            if -20 <= lon <= 55 and -35 <= lat <= 35:
                return 'Africa'
            return 'Other'

        src_cont = get_continent(source.coordinates[0], source.coordinates[1])
        dst_cont = get_continent(destination.coordinates[0], destination.coordinates[1])

        # If on different continents, force: inland -> port -> ship -> port -> inland
        if src_cont != dst_cont and 'Other' not in (src_cont, dst_cont):
            src_port = self.find_nearest_port(source)
            dst_port = self.find_nearest_port(destination)
            segments = []
            switchover_points = []
            # Truck to port
            truck1 = self.get_road_route(source, src_port, 'Truck')
            if truck1:
                segments.append(truck1)
                switchover_points.append(src_port)
            # Ship across ocean
            ship = self.get_maritime_route(src_port, dst_port)
            if ship:
                segments.append(ship)
                switchover_points.append(dst_port)
            # Truck to destination
            truck2 = self.get_road_route(dst_port, destination, 'Truck')
            if truck2:
                segments.append(truck2)
            total_distance = sum(seg.distance for seg in segments)
            total_duration = sum(seg.duration for seg in segments)
            total_co2 = sum(seg.co2_emissions for seg in segments)
            return RouteLeg(
                segments=segments,
                total_distance=total_distance,
                total_duration=total_duration,
                total_co2=total_co2,
                switchover_points=switchover_points
            )
        # Otherwise, use the default logic
        if intermediate_stops is None:
            intermediate_stops = []
        
        # Pre-process locations
        all_stops = [source] + intermediate_stops + [destination]
        processed_stops = []
        
        for stop in all_stops:
            if stop.type == 'inland':
                # Find nearest port or rail hub
                port = self.find_nearest_port(stop)
                rail_hub = self.find_nearest_rail_hub(stop)
                processed_stops.extend([stop, port, rail_hub])
            else:
                processed_stops.append(stop)
        
        # Plan segments
        segments = []
        switchover_points = []
        
        for i in range(len(processed_stops) - 1):
            current = processed_stops[i]
            next_stop = processed_stops[i + 1]
            
            # Determine appropriate mode
            if current.type == 'inland' and next_stop.type == 'port':
                segment = self.get_road_route(current, next_stop, 'Truck')
            elif current.type == 'port' and next_stop.type == 'port':
                segment = self.get_maritime_route(current, next_stop)
            elif current.type == 'port' and next_stop.type == 'inland':
                segment = self.get_road_route(current, next_stop, 'Truck')
            else:
                continue
            
            if segment:
                segments.append(segment)
                if i < len(processed_stops) - 2:
                    switchover_points.append(next_stop)
        
        # Calculate totals
        total_distance = sum(seg.distance for seg in segments)
        total_duration = sum(seg.duration for seg in segments)
        total_co2 = sum(seg.co2_emissions for seg in segments)
        
        return RouteLeg(
            segments=segments,
            total_distance=total_distance,
            total_duration=total_duration,
            total_co2=total_co2,
            switchover_points=switchover_points
        )

    def get_canal_waypoints(self, start, end):
        # Panama Canal
        panama_entry = (9.3, -79.9)
        panama_exit = (8.9, -79.6)
        # Suez Canal
        suez_entry = (30.6, 32.3)
        suez_exit = (30.0, 32.5)
        # Atlantic to Pacific (Americas to Asia/Australia) uses Panama
        if start[1] < -30 and end[1] > -30:
            return [panama_entry, panama_exit]
        # Mediterranean to Indian Ocean uses Suez
        if start[1] < 35 and end[1] > 35 and start[0] > 20 and end[0] < 30:
            return [suez_entry, suez_exit]
        return []

# Enhanced transport mode styling with minimal design and better readability
TRANSPORT_STYLES = {
    'Ship': {
        'color': '#FF6F00',  # vivid orange
        'weight': 3,  # Reduced weight for cleaner look
        'dash_array': '10, 20',  # Shorter dashes for minimal look
        'icon': 'ship',
        'prefix': 'fa',
        'label': 'Ship',
        'opacity': 0.85,
        'highlight_color': '#4169E1',
        'description': 'Maritime Transport'
    },
    'Electric Truck': {
        'color': '#00C853',  # vivid green
        'weight': 3,
        'dash_array': None,
        'icon': 'truck',
        'prefix': 'fa',
        'label': 'E-Truck',
        'opacity': 0.85,
        'highlight_color': '#228B22',
        'charging_stop_icon': 'bolt',
        'charging_stop_color': '#FFD700',
        'description': 'Electric Vehicle'
    },
    'Truck': {
        'color': '#FF8C00',  # Dark Orange
        'weight': 3,
        'dash_array': None,
        'icon': 'truck',
        'prefix': 'fa',
        'label': 'Truck',
        'opacity': 0.85,
        'highlight_color': '#FF4500',
        'description': 'Road Transport'
    },
    'Train': {
        'color': '#3949AB',  # deep blue
        'weight': 3,
        'dash_array': '5, 10',
        'icon': 'train',
        'prefix': 'fa',
        'label': 'Rail Transport',
        'opacity': 0.85,
        'highlight_color': '#8A2BE2',
        'description': 'Rail Transport'
    },
    'Plane': {
        'color': '#DC143C',  # Crimson
        'weight': 3,
        'dash_array': None,
        'icon': 'plane',
        'prefix': 'fa',
        'label': 'Air Transport',
        'opacity': 0.85,
        'highlight_color': '#B22222',
        'description': 'Air Transport'
    },
    'Hydrogen Truck': {
        'color': '#00CED1',  # Dark Turquoise
        'weight': 3,
        'dash_array': None,
        'icon': 'truck',
        'prefix': 'fa',
        'label': 'Hydrogen Truck',
        'opacity': 0.85,
        'highlight_color': '#008B8B',
        'description': 'Hydrogen Vehicle'
    },
    'Biofuel Truck': {
        'color': '#228B22',  # Forest Green
        'weight': 3,
        'dash_array': None,
        'icon': 'truck',
        'prefix': 'fa',
        'label': 'Biofuel Truck',
        'opacity': 0.85,
        'highlight_color': '#006400',
        'description': 'Biofuel Vehicle'
    }
}

# Major shipping lanes and waypoints
SHIPPING_LANES = {
    'panama_canal': {
        'name': 'Panama Canal Route',
        'waypoints': [
            (8.9, -79.6),  # Panama Canal Pacific Entrance
            (9.3, -79.9),  # Panama Canal Atlantic Entrance
        ]
    },
    'suez_canal': {
        'name': 'Suez Canal Route',
        'waypoints': [
            (30.6, 32.3),  # Suez Canal Mediterranean Entrance
            (30.0, 32.5),  # Suez Canal Red Sea Entrance
        ]
    },
    'cape_horn': {
        'name': 'Cape Horn Route',
        'waypoints': [
            (-55.9, -67.3),  # Cape Horn
            (-34.6, -58.4),  # Buenos Aires
        ]
    },
    'cape_good_hope': {
        'name': 'Cape of Good Hope Route',
        'waypoints': [
            (-34.4, 18.5),  # Cape of Good Hope
            (-33.9, 18.4),  # Cape Town
        ]
    }
}

# Major maritime regions and their boundaries
MARITIME_REGIONS = {
    'atlantic': {
        'name': 'Atlantic Ocean',
        'bounds': [(30, -80), (50, -20)],  # (lat, lon) pairs
    },
    'pacific': {
        'name': 'Pacific Ocean',
        'bounds': [(0, -180), (30, -120)],
    },
    'indian': {
        'name': 'Indian Ocean',
        'bounds': [(-30, 30), (0, 120)],
    }
}

# Maritime routing parameters
MARITIME_PARAMS = {
    'current_season': 'monsoon',  # Current season for route planning
    'hazard_buffer': 50,  # Buffer distance in km from known hazards
    'current_consideration': True,  # Consider ocean currents
    'weather_impact': True,  # Consider weather conditions
    'min_depth': 20,  # Minimum water depth in meters
    'max_wave_height': 5  # Maximum wave height in meters
}

# Electric truck parameters
ELECTRIC_TRUCK_PARAMS = {
    'max_range': 400,  # Maximum range in km
    'charging_time': 45,  # Charging time in minutes
    'min_charge_level': 0.2,  # Minimum charge level (20%)
    'max_charge_level': 0.9,  # Maximum charge level (90%)
    'charging_station_search_radius': 50  # Search radius in km for charging stations
}

def get_weather_data(lat: float, lon: float) -> Dict:
    """Get weather data from OpenWeatherMap API."""
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def get_maritime_traffic(lat: float, lon: float) -> List[Dict]:
    """Get maritime traffic data from MarineTraffic API."""
    api_key = os.getenv('MARINETRAFFIC_API_KEY')
    url = f"https://services.marinetraffic.com/api/exportvessels/v:8/MMSI:0/protocol:jsono/msgtype:simple/position:{lat},{lon}/radius:50/APIKey:{api_key}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else []

def is_point_in_region(point: Tuple[float, float], region: Dict) -> bool:
    """Check if a point is within a maritime region's bounds."""
    lat, lon = point
    (min_lat, min_lon), (max_lat, max_lon) = region['bounds']
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

def get_nearest_waypoint(point: Tuple[float, float], waypoints: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Find the nearest waypoint to a given point."""
    return min(waypoints, key=lambda wp: geodesic(point, wp).kilometers)

def get_maritime_route(origin: Tuple[float, float], destination: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Get maritime route coordinates using OpenSeaMap data and considering ocean currents and hazards.
    """
    try:
        # Calculate distance between points
        distance = geodesic(origin, destination).kilometers
        
        # For very short routes, use direct path
        if distance < 100:
            return [origin, destination]
        
        # Get initial route from ORS API
        coords = [[origin[1], origin[0]], [destination[1], destination[0]]]
        route = ors_client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson',
            instructions=False
        )
        
        # Extract coordinates from ORS response
        route_coords = [(coord[1], coord[0]) for coord in route['features'][0]['geometry']['coordinates']]
        
        # Check for major shipping lanes
        if (origin[1] < -80 and destination[1] > -80) or (origin[1] > -80 and destination[1] < -80):
            # Panama Canal route
            canal = SHIPPING_LANES['panama_canal']
            route_coords = []
            route_coords.extend(create_curved_route(origin, canal['waypoints'][0]))
            route_coords.extend(canal['waypoints'])
            route_coords.extend(create_curved_route(canal['waypoints'][1], destination))
        elif (origin[1] < 32 and destination[1] > 32) or (origin[1] > 32 and destination[1] < 32):
            # Suez Canal route
            canal = SHIPPING_LANES['suez_canal']
            route_coords = []
            route_coords.extend(create_curved_route(origin, canal['waypoints'][0]))
            route_coords.extend(canal['waypoints'])
            route_coords.extend(create_curved_route(canal['waypoints'][1], destination))
        
        # Add weather and traffic information to route points
        enhanced_route = []
        for i, point in enumerate(route_coords):
            # Get weather data
            weather = get_weather_data(point[0], point[1])
            if weather:
                wind_speed = weather['wind']['speed']
                # Adjust route based on wind conditions
                if wind_speed > 10:  # Strong winds
                    # Add slight deviation to avoid strong winds
                    deviation = 0.1 * (1 if i % 2 == 0 else -1)
                    point = (point[0] + deviation, point[1] + deviation)
            
            # Get maritime traffic
            traffic = get_maritime_traffic(point[0], point[1])
            if traffic:
                # Add slight deviation if high traffic
                if len(traffic) > 5:
                    deviation = 0.05 * (1 if i % 2 == 0 else -1)
                    point = (point[0] + deviation, point[1] + deviation)
            
            enhanced_route.append(point)
        
        # Smooth the route
        return smooth_route(enhanced_route)
        
    except Exception as e:
        print(f"Error calculating maritime route: {e}")
        return [origin, destination]

def smooth_route(route: List[Tuple[float, float]], smoothing_factor: float = 0.3) -> List[Tuple[float, float]]:
    """Smooth a route using moving average."""
    if len(route) <= 2:
        return route
    
    smoothed = [route[0]]
    for i in range(1, len(route) - 1):
        prev = route[i - 1]
        curr = route[i]
        next_point = route[i + 1]
        
        # Calculate smoothed point
        smoothed_lat = curr[0] + smoothing_factor * (
            (prev[0] + next_point[0]) / 2 - curr[0]
        )
        smoothed_lon = curr[1] + smoothing_factor * (
            (prev[1] + next_point[1]) / 2 - curr[1]
        )
        
        smoothed.append((smoothed_lat, smoothed_lon))
    
    smoothed.append(route[-1])
    return smoothed

def create_curved_route(start: Tuple[float, float], end: Tuple[float, float], num_points: int = 10) -> List[Tuple[float, float]]:
    """Create a curved route between two points."""
    points = []
    for i in range(num_points + 1):
        ratio = i / num_points
        # Add a natural curve using sine wave
        curve = 2 * math.sin(ratio * math.pi)  # Natural curve
        lat = start[0] + (end[0] - start[0]) * ratio + curve
        lon = start[1] + (end[1] - start[1]) * ratio
        points.append((lat, lon))
    return points

def get_route_coordinates(origin: Tuple[float, float], destination: Tuple[float, float], mode: str) -> List[Tuple[float, float]]:
    """
    Get route coordinates using ORS API for accurate road routing with enhanced parameters.
    """
    try:
        if mode == 'Ship':
            return get_maritime_route(origin, destination)
        else:
            coords = [[origin[1], origin[0]], [destination[1], destination[0]]]
            
            if mode in ['Truck', 'Electric Truck', 'Hydrogen Truck', 'Biofuel Truck']:
                profile = 'driving-hgv'
                # Enhanced parameters for truck routing with better road following
                params = {
                    'preference': 'fastest',
                    'geometry_simplify': False,
                    'continue_straight': False,
                    'optimize_waypoints': True,
                    'avoid_features': ['highways', 'tollways'],
                    'vehicle_type': 'hgv',
                    'weight': 3.5,
                    'height': 4.0,
                    'width': 2.5,
                    'length': 12.0,
                    'axleload': 11.5,
                    'avoid_borders': 'controlled',
                    'avoid_countries': [],
                    'avoid_polygons': [],
                    'ch.disable': True,
                    'instructions': True,
                    'roundabout_exits': True,
                    'attributes': ['avgspeed', 'steepness', 'waytype', 'surface'],
                    'elevation': True,  # Consider elevation for better routing
                    'radiuses': [-1, -1],  # Search radius for start/end points
                    'bearings': [[0, 360], [0, 360]],  # Allow all directions
                    'skip_segments': [],  # Don't skip any segments
                    'alternative_routes': {
                        'target_count': 1,
                        'weight_factor': 1.4
                    }
                }
            else:
                profile = 'driving-car'
                params = {
                    'preference': 'fastest',
                    'geometry_simplify': False,
                    'continue_straight': False,
                    'optimize_waypoints': True
                }
            
            # Get detailed route with full geometry and instructions
            route = ors_client.directions(
                coordinates=coords,
                profile=profile,
                format='geojson',
                instructions=True,
                **params
            )
            
            if route and 'features' in route and len(route['features']) > 0:
                coordinates = [(coord[1], coord[0]) for coord in route['features'][0]['geometry']['coordinates']]
                instructions = route['features'][0]['properties']['segments'][0]['steps']
                
                # Enhanced route with detailed turn points and road following
                enhanced_coordinates = []
                for i, instruction in enumerate(instructions):
                    # Add the start point of each instruction
                    start_coord = (instruction['location'][1], instruction['location'][0])
                    enhanced_coordinates.append(start_coord)
                    
                    # Add intermediate points for better road following
                    if 'geometry' in instruction:
                        for coord in instruction['geometry']['coordinates']:
                            enhanced_coordinates.append((coord[1], coord[0]))
                    
                    # Add extra points for sharp turns to ensure road following
                    if i < len(instructions) - 1:
                        next_instruction = instructions[i + 1]
                        if 'maneuver' in next_instruction and next_instruction['maneuver'].get('type') in ['turn', 'roundabout']:
                            # Add intermediate points for smoother turns
                            mid_point = (
                                (start_coord[0] + next_instruction['location'][1]) / 2,
                                (start_coord[1] + next_instruction['location'][0]) / 2
                            )
                            enhanced_coordinates.append(mid_point)
                
                # Add the final destination point
                enhanced_coordinates.append(coordinates[-1])
                
                return enhanced_coordinates
            
            return [origin, destination]
            
    except Exception as e:
        print(f"Error calculating route: {e}")
        return [origin, destination]

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Calculate the angle between three points in degrees.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Calculate magnitudes
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Calculate angle in radians
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(min(cos_angle, 1), -1)  # Clamp to [-1, 1]
    angle_rad = math.acos(cos_angle)
    
    # Convert to degrees
    return math.degrees(angle_rad)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def render_emission_map(emissions: pd.DataFrame, _get_coords_func: Callable, max_routes: int = 5) -> folium.Map:
    """Render an emission map with caching."""
    try:
        # Create base map
        center_lat = emissions['source_lat'].mean() if 'source_lat' in emissions else 0
        center_lon = emissions['source_lon'].mean() if 'source_lon' in emissions else 0
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Add routes with progress bar
        progress_bar = st.progress(0)
        total_routes = min(len(emissions), max_routes)
        
        for idx, row in emissions.head(max_routes).iterrows():
            try:
                # Update progress
                progress = (idx + 1) / total_routes
                progress_bar.progress(progress)
                
                # Get coordinates
                source_coords = _get_coords_func(row['source_country'], row['source_city'])
                dest_coords = _get_coords_func(row['dest_country'], row['dest_city'])
                
                if not source_coords or not dest_coords:
                    continue
                
                # Debug output for the first route
                if idx == 0:
                    st.write(f"Drawing route from {row['source']} ({source_coords}) to {row['destination']} ({dest_coords})")
                
                # Add route to map
                folium.PolyLine(
                    locations=[source_coords, dest_coords],
                    color='red',
                    weight=2,
                    opacity=0.8,
                    popup=f"CO2: {row['co2_kg']:.2f} kg"
                ).add_to(m)
                
                # Add markers
                folium.Marker(
                    source_coords,
                    popup=f"Source: {row['source']}",
                    icon=folium.Icon(color='green')
                ).add_to(m)
                
                folium.Marker(
                    dest_coords,
                    popup=f"Destination: {row['destination']}",
                    icon=folium.Icon(color='red')
                ).add_to(m)
                
            except Exception as e:
                logger.error(f"Error rendering route {idx}: {e}")
                continue
        
        return m
        
    except Exception as e:
        logger.error(f"Error rendering emission map: {e}")
        raise
    finally:
        # Clear progress bar
        if 'progress_bar' in locals():
            progress_bar.empty()

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    """
    Create a bar chart using Plotly.
    """
    fig = px.bar(df, x=x, y=y, title=title, color=color) if color else px.bar(df, x=x, y=y, title=title)
    return fig

def pie_chart(df: pd.DataFrame, values: str, names: str, title: str) -> go.Figure:
    """
    Create a pie chart using Plotly.
    """
    fig = px.pie(df, values=values, names=names, title=title)
    return fig

def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """
    Create a line chart using Plotly.
    """
    fig = px.line(df, x=x, y=y, title=title)
    return fig

def gauge_chart(value: float, title: str, max_value: float = 100) -> go.Figure:
    """
    Create a gauge chart for KPIs.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range': [0, max_value]}, 'bar': {'color': "#36A2EB"}}
    ))
    return fig 

def create_custom_icon(icon_name: str, color: str, size: int = 16, icon_type: str = 'point') -> DivIcon:
    """Create a minimal custom icon with Font Awesome."""
    if icon_type == 'point':
        # Minimal point markers with subtle background
        return DivIcon(
            html=f'''
            <div style="
                background-color: rgba(255, 255, 255, 0.8);
                border: 2px solid {color};
                border-radius: 50%;
                width: {size}px;
                height: {size}px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <i class="fa fa-{icon_name}" style="
                    color: {color};
                    font-size: {size * 0.6}px;
                    line-height: 1;
                "></i>
            </div>
            ''',
            icon_size=(size, size),
            icon_anchor=(size/2, size/2)
        )
    else:
        # Regular icon for other elements
        return DivIcon(
            html=f'<i class="fa fa-{icon_name}" style="color: {color}; font-size: {size}px;"></i>',
            icon_size=(size, size),
            icon_anchor=(size/2, size/2)
        )

def get_electric_truck_route(origin: Tuple[float, float], destination: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Get electric truck route with charging stops based on range and battery capacity.
    """
    try:
        # Get initial route from ORS API
        coords = [[origin[1], origin[0]], [destination[1], destination[0]]]
        route = ors_client.directions(
            coordinates=coords,
            profile='driving-hgv',
            format='geojson',
            instructions=False,
            **{
                'preference': 'fastest',
                'geometry_simplify': False,
                'continue_straight': True,
                'optimize_waypoints': True,
                'avoid_features': ['highways', 'tollways'],
                'vehicle_type': 'hgv',
                'weight': 3.5,
                'height': 4.0,
                'width': 2.5,
                'length': 12.0,
                'axleload': 11.5,
            }
        )
        
        if route and 'features' in route and len(route['features']) > 0:
            coordinates = [(coord[1], coord[0]) for coord in route['features'][0]['geometry']['coordinates']]
            
            # Calculate total distance
            total_distance = 0
            for i in range(len(coordinates) - 1):
                total_distance += geodesic(coordinates[i], coordinates[i + 1]).kilometers
            
            # Calculate number of charging stops needed
            num_stops = math.ceil(total_distance / ELECTRIC_TRUCK_PARAMS['max_range'])
            
            # Add charging stops
            if num_stops > 1:
                enhanced_coordinates = []
                for i in range(len(coordinates) - 1):
                    enhanced_coordinates.append(coordinates[i])
                    
                    # Add charging stop if needed
                    if (i + 1) % (len(coordinates) // num_stops) == 0:
                        # Find nearest charging station
                        mid_point = coordinates[i]
                        charging_station = find_nearest_charging_station(mid_point)
                        if charging_station:
                            enhanced_coordinates.append(charging_station)
                
                enhanced_coordinates.append(coordinates[-1])
                return enhanced_coordinates
            
            return coordinates
        
        return [origin, destination]
        
    except Exception as e:
        print(f"Error calculating electric truck route: {e}")
        return [origin, destination]

def find_nearest_charging_station(point: Tuple[float, float]) -> Tuple[float, float]:
    """
    Find the nearest charging station to a given point.
    This is a placeholder function - in a real implementation, you would query a charging station database.
    """
    # For now, return the point itself as we don't have a real charging station database
    return point

# --- Universal multi-modal route planner and visualizer ---
def plan_multi_modal_route(source_coords, dest_coords):
    # 1. Find nearest road hub to source
    road_start = find_nearest_waypoint_mode(*source_coords, ROAD_WAYPOINTS)
    road_start_coords = ROAD_WAYPOINTS[road_start]['coords']
    # 2. Find nearest port to end of road leg (from road hub)
    port_us = find_nearest_waypoint_mode(*road_start_coords, WAYPOINTS)
    port_us_coords = WAYPOINTS[port_us]['coords']
    # 3. Find nearest port to destination
    port_eu = find_nearest_waypoint_mode(*dest_coords, WAYPOINTS)
    port_eu_coords = WAYPOINTS[port_eu]['coords']
    # 4. Find nearest rail hub to destination (from port)
    rail_end = find_nearest_waypoint_mode(*port_eu_coords, RAIL_WAYPOINTS)
    rail_end_coords = RAIL_WAYPOINTS[rail_end]['coords']

    # 5. Build each leg
    road_leg = generate_mode_path(source_coords, port_us_coords, ROAD_WAYPOINTS)
    ship_leg = generate_mode_path(port_us_coords, port_eu_coords, WAYPOINTS)
    rail_leg = generate_mode_path(port_eu_coords, dest_coords, RAIL_WAYPOINTS)

    # 6. Return all legs and switchover points
    return [
        {'mode': 'Electric Truck', 'path': road_leg, 'start': source_coords, 'end': port_us_coords, 'switchover': port_us_coords},
        {'mode': 'Ship', 'path': ship_leg, 'start': port_us_coords, 'end': port_eu_coords, 'switchover': port_eu_coords},
        {'mode': 'Rail', 'path': rail_leg, 'start': port_eu_coords, 'end': dest_coords, 'switchover': dest_coords}
    ]

# --- Update render_multi_modal_route to use the new planner ---
def render_multi_modal_route(route_data: dict, get_coords_func) -> folium.Map:
    try:
        source_coords = get_coords_func(route_data['source']['country'], route_data['source']['city'])
        dest_coords = get_coords_func(route_data['destination']['country'], route_data['destination']['city'])
        m = folium.Map(
            location=[(source_coords[0] + dest_coords[0]) / 2, (source_coords[1] + dest_coords[1]) / 2],
            zoom_start=3,
            tiles='OpenStreetMap',
            control_scale=True
        )
        m.add_child(MeasureControl())

        # Plan the multi-modal route
        legs = plan_multi_modal_route(source_coords, dest_coords)

        # Add legend
        legend_html = '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; background: rgba(255,255,255,0.95); border: 1px solid #ddd; border-radius: 6px; padding: 10px; font-size: 13px; font-family: Arial, sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
        legend_html += '<div style="font-weight: bold; margin-bottom: 6px;">Legend</div>'
        for mode in ['Electric Truck', 'Ship', 'Rail']:
            style = TRANSPORT_STYLES[mode]
            dash = 'border-bottom:2px dashed %s;' % style['color'] if style.get('dash_array') else ''
            legend_html += f'<div style="margin-bottom: 4px;"><span style="display:inline-block;width:18px;height:4px;background:{style["color"]};border-radius:2px;margin-right:8px;{dash}"></span> {style["label"]}</div>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

        # Draw each leg
        for i, leg in enumerate(legs):
            # Defensive: skip if mode not in styles
            if 'mode' not in leg or leg['mode'] not in TRANSPORT_STYLES:
                logger.error(f"Mode {leg.get('mode', None)} not in TRANSPORT_STYLES")
                continue
            style = TRANSPORT_STYLES[leg['mode']]
            # Defensive: skip if path is invalid (less than 2 points or all points identical)
            if 'path' not in leg or not leg['path'] or len(leg['path']) < 2 or all(p == leg['path'][0] for p in leg['path']):
                logger.error(f"Leg {i} ({leg.get('mode', None)}) has invalid path: {leg.get('path', None)}")
                continue

            # Draw the route
            if leg['mode'] == 'Ship':
                AntPath(
                    locations=leg['path'],
                    color=style['color'],
                    weight=5,
                    dash_array=[10, 20],
                    pulse_color='#FFD700',
                    delay=800,
                    opacity=0.9,
                    reverse=False,
                    tooltip=f"{leg['mode']} Route"
                ).add_to(m)
            else:
                folium.PolyLine(
                    locations=leg['path'],
                    color=style['color'],
                    weight=5,
                    dash_array=style.get('dash_array'),
                    tooltip=f"{leg['mode']} Route"
                ).add_to(m)

            # Add start marker for first leg
            if i == 0:
                folium.Marker(
                    location=leg['start'],
                    tooltip=f"Start: {leg['mode']}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)

            # Add switchover marker (except for last leg, which is the destination)
            if i < len(legs) - 1:
                folium.Marker(
                    location=leg['end'],
                    tooltip=f"Switchover: {legs[i+1]['mode']}",
                    icon=folium.Icon(color='blue', icon='exchange')
                ).add_to(m)

            # Add end marker for last leg
            if i == len(legs) - 1:
                folium.Marker(
                    location=leg['end'],
                    tooltip=f"End: {leg['mode']}",
                    icon=folium.Icon(color='orange', icon='flag')
                ).add_to(m)

        # Fallback: if no legs or all paths are empty, draw direct line
        if not legs or all(len(leg['path']) < 2 for leg in legs):
            folium.PolyLine(
                locations=[source_coords, dest_coords],
                color='black',
                weight=4,
                dash_array='5,10',
                tooltip='Direct Route (Fallback)'
            ).add_to(m)

        m.fit_bounds([source_coords, dest_coords])
        return m

    except Exception as e:
        logger.error(f"Exception in render_multi_modal_route: {e}")
        error_map = folium.Map(location=[0, 0], zoom_start=2)
        folium.Popup(f"Error loading route: {str(e)}", max_width=300).add_to(error_map)
        return error_map

@dataclass
class TruckStop:
    """Represents a stop point for trucks (charging, rest, or mandatory)."""
    location: Tuple[float, float]
    name: str
    type: str  # 'charging', 'rest', 'mandatory'
    duration: float  # in hours
    charge_level: Optional[float] = None  # for charging stops (0.0 to 1.0)
    distance_from_start: float = 0.0  # in kilometers

@dataclass
class TruckSegment:
    """Represents a segment of the truck route with stops and constraints."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    path: List[Tuple[float, float]]
    distance: float  # in kilometers
    duration: float  # in hours
    stops: List[TruckStop]
    road_type: str  # 'highway', 'local', etc.
    speed_limit: float  # in km/h
    weather_conditions: Dict[str, Any]
    traffic_level: str  # 'low', 'medium', 'high'

class RoadSegment:
    """Represents a segment of road with its properties."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    road_type: str  # 'highway', 'arterial', 'local'
    distance: float  # in kilometers
    base_time: float  # in hours
    truck_allowed: bool
    height_limit: float  # in meters
    weight_limit: float  # in tons
    has_charging: bool  # for electric trucks
    has_fuel: bool  # for diesel trucks
    traffic_level: str  # 'low', 'medium', 'high'
    weather_impact: float  # multiplier for travel time
    is_closed: bool

class TruckConstraints:
    """Represents constraints for a specific truck type."""
    max_range: float  # in kilometers
    charging_time: float  # in hours
    min_charge_level: float  # 0.0 to 1.0
    max_charge_level: float  # 0.0 to 1.0
    max_daily_hours: float  # in hours
    rest_time: float  # in hours
    height: float  # in meters
    weight: float  # in tons
    avg_speed: float  # in km/h
    co2_per_km: float  # in kg/km

class RoadNetwork:
    """Represents the road network with real-time data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the road network with API keys and clients."""
        self.ors_client = None
        if api_key:
            try:
                self.ors_client = client.Client(key=api_key, timeout=5)
            except Exception as e:
                logger.error(f"Failed to initialize ORS client: {e}")
        
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        if not self.ors_client:
            logger.warning("OpenRouteService client not configured. Some features will be disabled.")
        if not self.weather_api_key:
            logger.warning("Weather API key not found. Weather features will be disabled.")
        if not self.google_maps_api_key:
            logger.warning("Google Maps API key not found. Some map features will be disabled.")

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _load_road_network(self, bounds: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Load road network data with caching."""
        if not self.ors_client:
            raise ValueError("OpenRouteService client not configured")
        
        try:
            return _cached_api_call(
                "https://api.openrouteservice.org/v2/directions/driving-car",
                {
                    "start": f"{bounds[0]},{bounds[1]}",
                    "end": f"{bounds[2]},{bounds[3]}",
                    "api_key": self.ors_client.key
                }
            )
        except Exception as e:
            logger.error(f"Failed to load road network: {e}")
            raise

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _update_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Update weather data with caching."""
        if not self.weather_api_key:
            return {}
        
        try:
            return _cached_api_call(
                "https://api.openweathermap.org/data/2.5/weather",
                {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.weather_api_key,
                    "units": "metric"
                }
            )
        except Exception as e:
            logger.error(f"Failed to update weather data: {e}")
            return {}
    
    def update_road_conditions(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        """Update road conditions with real-time data."""
        try:
            # Update traffic data
            self._update_traffic_data(bounds)
            # Update weather data
            self._update_weather_data(bounds)
            # Update road closures
            self._update_road_closures(bounds)
        except Exception as e:
            logger.error(f"Error updating road conditions: {e}", exc_info=True)
    
    def _update_traffic_data(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        """Update traffic data from Google Maps API."""
        try:
            # This would use Google Maps API to get real-time traffic
            # For now, we'll use mock data
            for segment_id, segment in self.segments.items():
                hour = datetime.now().hour
                if 7 <= hour <= 9 or 16 <= hour <= 18:
                    segment.traffic_level = 'high'
                    segment.base_time *= 1.5
                elif 10 <= hour <= 15:
                    segment.traffic_level = 'medium'
                    segment.base_time *= 1.2
                else:
                    segment.traffic_level = 'low'
        except Exception as e:
            print(f"Error updating traffic data: {e}")
    
    def _update_road_closures(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        """Update road closure information."""
        try:
            # This would typically use a road closure API
            # For now, we'll use mock data
            for segment_id, segment in self.segments.items():
                # Randomly mark some segments as closed (for testing)
                if random.random() < 0.01:  # 1% chance of closure
                    segment.is_closed = True
                    logging.info(f"Road segment {segment_id} marked as closed for testing")
        except Exception as e:
            logging.error(f"Error updating road closures: {e}", exc_info=True)

class TruckRouter:
    """Handles truck route planning with real-world constraints using Dijkstra's algorithm."""
    
    def __init__(self):
        self.ors_client = openrouteservice.Client(key=os.getenv('ORS_API_KEY'))
        self.weather_api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        self.traffic_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        # Truck-specific parameters with enhanced constraints
        self.truck_params = {
            'electric': {
                'max_range': 400,  # km
                'charging_time': 1.0,  # hours
                'min_charge_level': 0.2,  # 20%
                'max_charge_level': 0.9,  # 90%
                'avg_speed': 60,  # km/h
                'max_daily_hours': 11,  # hours
                'rest_time': 11,  # hours
                'co2_per_km': 0.0,  # kg/km (electric)
                'height': 4.0,  # meters
                'weight': 3.5,  # tons
                'preferred_roads': ['highway', 'trunk', 'primary'],
                'avoid_roads': ['residential', 'service', 'track']
            },
            'diesel': {
                'max_range': 1000,  # km
                'refuel_time': 0.5,  # hours
                'min_fuel_level': 0.1,  # 10%
                'max_fuel_level': 0.9,  # 90%
                'avg_speed': 65,  # km/h
                'max_daily_hours': 11,  # hours
                'rest_time': 11,  # hours
                'co2_per_km': 0.8,  # kg/km
                'height': 4.0,  # meters
                'weight': 3.5,  # tons
                'preferred_roads': ['highway', 'trunk', 'primary'],
                'avoid_roads': ['residential', 'service', 'track']
            }
        }
    
    def get_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                 truck_type: str = 'electric') -> Dict[str, Any]:
        """
        Get a detailed truck route using Dijkstra's algorithm with real-time data.
        """
        try:
            # Get base route from ORS API with enhanced parameters
            coords = [[start[1], start[0]], [end[1], end[0]]]
            params = {
                'preference': 'fastest',
                'geometry_simplify': False,
                'continue_straight': False,
                'optimize_waypoints': True,
                'avoid_features': self.truck_params[truck_type]['avoid_roads'],
                'vehicle_type': 'hgv',
                'weight': self.truck_params[truck_type]['weight'],
                'height': self.truck_params[truck_type]['height'],
                'width': 2.5,
                'length': 12.0,
                'axleload': 11.5,
                'elevation': True,
                'instructions': True,
                'attributes': ['avgspeed', 'steepness', 'waytype', 'surface', 'tollways', 'highways'],
                'radiuses': [-1, -1],  # Search radius for start/end points
                'bearings': [[0, 360], [0, 360]],  # Allow all directions
                'alternative_routes': {
                    'target_count': 3,  # Get alternative routes
                    'weight_factor': 1.4
                }
            }
            
            # Get route with alternatives
            route = self.ors_client.directions(
                coordinates=coords,
                profile='driving-hgv',
                format='geojson',
                **params
            )
            
            if not route or 'features' not in route or len(route['features']) == 0:
                raise ValueError("No route found")
            
            # Select best route based on Dijkstra's algorithm
            best_route = self._select_best_route(route['features'], truck_type)
            
            # Extract route details
            path = [(coord[1], coord[0]) for coord in best_route['geometry']['coordinates']]
            instructions = best_route['properties']['segments'][0]['steps']
            
            # Get truck parameters
            params = self.truck_params[truck_type]
            
            # Segment the route and add stops using Dijkstra's algorithm
            segments = []
            current_segment = []
            current_distance = 0
            stops = []
            total_distance = 0
            total_duration = 0
            daily_duration = 0
            
            for i, instruction in enumerate(instructions):
                # Get instruction details
                step_distance = instruction['distance'] / 1000  # Convert to km
                step_duration = instruction['duration'] / 3600  # Convert to hours
                step_coords = [(coord[1], coord[0]) for coord in instruction['geometry']['coordinates']]
                
                # Get road type and restrictions
                road_type = instruction.get('waytype', 'unknown')
                speed_limit = instruction.get('avgspeed', params['avg_speed'])
                
                # Check road restrictions
                if road_type in params['avoid_roads']:
                    # Find alternative route around this segment
                    alt_route = self._find_alternative_route(
                        step_coords[0], 
                        step_coords[-1],
                        params['preferred_roads']
                    )
                    if alt_route:
                        step_coords = alt_route
                
                # Add step to current segment
                current_segment.extend(step_coords)
                current_distance += step_distance
                total_distance += step_distance
                daily_duration += step_duration
                
                # Check if we need a stop (using Dijkstra's algorithm for optimal stop placement)
                if current_distance >= params['max_range'] * 0.8:  # Stop at 80% of max range
                    # Find optimal stop location using Dijkstra's algorithm
                    stop_location = self._find_optimal_stop_location(
                        current_segment[-1],
                        params['max_range'] * 0.2,  # 20% range buffer
                        truck_type
                    )
                    
                    if stop_location:
                        stop = TruckStop(
                            location=stop_location,
                            name=f"Stop {len(stops) + 1}",
                            type='charging' if truck_type == 'electric' else 'refuel',
                            duration=params['charging_time'] if truck_type == 'electric' else params['refuel_time'],
                            charge_level=params['max_charge_level'],
                            distance_from_start=total_distance
                        )
                        stops.append(stop)
                        
                        # Create segment up to this stop
                        segments.append(TruckSegment(
                            start=current_segment[0],
                            end=stop_location,
                            path=current_segment,
                            distance=current_distance,
                            duration=current_distance / params['avg_speed'],
                            stops=[stop],
                            road_type=road_type,
                            speed_limit=speed_limit,
                            weather_conditions=self._get_weather_conditions(stop_location),
                            traffic_level=self._get_traffic_level(stop_location)
                        ))
                        
                        # Start new segment
                        current_segment = [stop_location]
                        current_distance = 0
                
                # Check for mandatory rest stops (every 11 hours)
                if daily_duration >= params['max_daily_hours']:
                    # Find optimal rest stop location
                    rest_location = self._find_optimal_rest_stop(
                        step_coords[-1],
                        params['preferred_roads']
                    )
                    
                    rest_stop = TruckStop(
                        location=rest_location,
                        name=f"Rest Stop {len(stops) + 1}",
                        type='rest',
                        duration=params['rest_time'],
                        distance_from_start=total_distance
                    )
                    stops.append(rest_stop)
                    daily_duration = 0  # Reset daily driving time
            
            # Add final segment if needed
            if current_segment:
                segments.append(TruckSegment(
                    start=current_segment[0],
                    end=end,
                    path=current_segment,
                    distance=current_distance,
                    duration=current_distance / params['avg_speed'],
                    stops=[],
                    road_type=instructions[-1].get('waytype', 'unknown'),
                    speed_limit=instructions[-1].get('avgspeed', params['avg_speed']),
                    weather_conditions=self._get_weather_conditions(end),
                    traffic_level=self._get_traffic_level(end)
                ))
            
            # Calculate total duration including stops
            total_duration = sum(seg.duration for seg in segments)
            total_duration += sum(stop.duration for stop in stops)
            
            return {
                'segments': segments,
                'stops': stops,
                'total_distance': total_distance,
                'total_duration': total_duration,
                'truck_type': truck_type,
                'co2_emissions': total_distance * params['co2_per_km']
            }
            
        except Exception as e:
            print(f"Error calculating truck route: {e}")
            return None
    
    def _select_best_route(self, routes: List[Dict], truck_type: str) -> Dict:
        """Select best route using Dijkstra's algorithm with truck constraints."""
        best_route = None
        best_weight = float('inf')
        
        for route in routes:
            weight = 0
            for segment in route['properties']['segments'][0]['steps']:
                # Calculate segment weight based on multiple factors
                distance = segment['distance'] / 1000  # km
                duration = segment['duration'] / 3600  # hours
                road_type = segment.get('waytype', 'unknown')
                
                # Base weight is travel time
                segment_weight = duration
                
                # Adjust for road type
                if road_type in self.truck_params[truck_type]['preferred_roads']:
                    segment_weight *= 0.8  # Prefer highways
                elif road_type in self.truck_params[truck_type]['avoid_roads']:
                    segment_weight *= 1.5  # Avoid residential roads
                
                # Adjust for traffic
                traffic = self._get_traffic_level((
                    segment['location'][1],
                    segment['location'][0]
                ))
                if traffic == 'high':
                    segment_weight *= 1.5
                elif traffic == 'medium':
                    segment_weight *= 1.2
                
                # Adjust for weather
                weather = self._get_weather_conditions((
                    segment['location'][1],
                    segment['location'][0]
                ))
                if weather.get('weather') in ['Rain', 'Snow', 'Thunderstorm']:
                    segment_weight *= 1.3
                elif weather.get('weather') in ['Mist', 'Fog']:
                    segment_weight *= 1.2
                
                weight += segment_weight
            
            if weight < best_weight:
                best_weight = weight
                best_route = route
        
        return best_route
    
    def _find_alternative_route(self, start: Tuple[float, float], end: Tuple[float, float],
                              preferred_roads: List[str]) -> Optional[List[Tuple[float, float]]]:
        """Find alternative route avoiding restricted roads."""
        try:
            coords = [[start[1], start[0]], [end[1], end[0]]]
            params = {
                'preference': 'fastest',
                'geometry_simplify': False,
                'continue_straight': False,
                'optimize_waypoints': True,
                'avoid_features': ['highways', 'tollways'],
                'vehicle_type': 'hgv',
                'instructions': False
            }
            
            route = self.ors_client.directions(
                coordinates=coords,
                profile='driving-hgv',
                format='geojson',
                **params
            )
            
            if route and 'features' in route and len(route['features']) > 0:
                return [(coord[1], coord[0]) for coord in route['features'][0]['geometry']['coordinates']]
            
        except Exception as e:
            print(f"Error finding alternative route: {e}")
        
        return None
    
    def _find_optimal_stop_location(self, point: Tuple[float, float], 
                                  max_distance: float, truck_type: str) -> Optional[Tuple[float, float]]:
        """Find optimal stop location using Dijkstra's algorithm."""
        try:
            # Search for charging/fuel stations within range
            coords = [[point[1], point[0]]]
            params = {
                'radius': max_distance * 1000,  # Convert to meters
                'limit': 5,
                'category': 'charging_station' if truck_type == 'electric' else 'fuel_station'
            }
            
            response = self.ors_client.pois(
                request='pois',
                coordinates=coords,
                **params
            )
            
            if response and 'features' in response and len(response['features']) > 0:
                # Find closest station that's on a preferred road
                best_station = None
                best_weight = float('inf')
                
                for station in response['features']:
                    station_coords = (station['geometry']['coordinates'][1],
                                    station['geometry']['coordinates'][0])
                    
                    # Calculate weight based on distance and road type
                    distance = geodesic(point, station_coords).kilometers
                    road_type = station['properties'].get('waytype', 'unknown')
                    
                    weight = distance
                    if road_type in self.truck_params[truck_type]['preferred_roads']:
                        weight *= 0.8  # Prefer stations on highways
                    
                    if weight < best_weight:
                        best_weight = weight
                        best_station = station_coords
                
                return best_station
            
        except Exception as e:
            print(f"Error finding optimal stop location: {e}")
        
        return point  # Fallback to original point
    
    def _find_optimal_rest_stop(self, point: Tuple[float, float],
                              preferred_roads: List[str]) -> Tuple[float, float]:
        """Find optimal rest stop location."""
        try:
            # Search for truck stops or rest areas
            coords = [[point[1], point[0]]]
            params = {
                'radius': 5000,  # 5km radius
                'limit': 3,
                'category': 'truck_stop'
            }
            
            response = self.ors_client.pois(
                request='pois',
                coordinates=coords,
                **params
            )
            
            if response and 'features' in response and len(response['features']) > 0:
                # Find closest stop on a preferred road
                best_stop = None
                best_weight = float('inf')
                
                for stop in response['features']:
                    stop_coords = (stop['geometry']['coordinates'][1],
                                 stop['geometry']['coordinates'][0])
                    
                    # Calculate weight based on distance and road type
                    distance = geodesic(point, stop_coords).kilometers
                    road_type = stop['properties'].get('waytype', 'unknown')
                    
                    weight = distance
                    if road_type in preferred_roads:
                        weight *= 0.8  # Prefer stops on highways
                    
                    if weight < best_weight:
                        best_weight = weight
                        best_stop = stop_coords
                
                return best_stop if best_stop else point
            
        except Exception as e:
            print(f"Error finding optimal rest stop: {e}")
        
        return point  # Fallback to original point

# --- Waypoint networks for each mode ---
# Maritime: WAYPOINTS (already defined)
# Road (electric truck):
ROAD_WAYPOINTS = {
    "Chicago": {"coords": (41.8781, -87.6298), "neighbors": ["New York", "Los Angeles"]},
    "New York": {"coords": (40.7128, -74.0060), "neighbors": ["Chicago", "Hamburg"]},
    "Los Angeles": {"coords": (34.0522, -118.2437), "neighbors": ["Chicago"]},
    "Hamburg": {"coords": (53.5511, 9.9937), "neighbors": ["Berlin", "Rotterdam", "New York"]},
    "Berlin": {"coords": (52.5200, 13.4050), "neighbors": ["Hamburg", "Munich"]},
    "Munich": {"coords": (48.1351, 11.5820), "neighbors": ["Berlin", "Zurich"]},
    "Zurich": {"coords": (47.3769, 8.5417), "neighbors": ["Munich", "Milan"]},
    "Milan": {"coords": (45.4642, 9.1900), "neighbors": ["Zurich", "Rome"]},
    "Rome": {"coords": (41.9028, 12.4964), "neighbors": ["Milan", "Naples"]},
    "Naples": {"coords": (40.8518, 14.2681), "neighbors": ["Rome"]},
    "Rotterdam": {"coords": (51.9244, 4.4777), "neighbors": ["Hamburg", "Paris"]},
    "Paris": {"coords": (48.8566, 2.3522), "neighbors": ["Rotterdam", "Lyon"]},
    "Lyon": {"coords": (45.7640, 4.8357), "neighbors": ["Paris", "Marseille"]},
    "Marseille": {"coords": (43.2965, 5.3698), "neighbors": ["Lyon"]},
    # Add more for realism
}
# Rail:
RAIL_WAYPOINTS = {
    "Chicago": {"coords": (41.8781, -87.6298), "neighbors": ["New York", "Los Angeles"]},
    "New York": {"coords": (40.7128, -74.0060), "neighbors": ["Chicago", "Hamburg"]},
    "Los Angeles": {"coords": (34.0522, -118.2437), "neighbors": ["Chicago"]},
    "Hamburg": {"coords": (53.5511, 9.9937), "neighbors": ["Berlin", "Frankfurt", "New York"]},
    "Berlin": {"coords": (52.5200, 13.4050), "neighbors": ["Hamburg", "Warsaw"]},
    "Frankfurt": {"coords": (50.1109, 8.6821), "neighbors": ["Hamburg", "Zurich"]},
    "Zurich": {"coords": (47.3769, 8.5417), "neighbors": ["Frankfurt", "Milan"]},
    "Milan": {"coords": (45.4642, 9.1900), "neighbors": ["Zurich", "Rome"]},
    "Rome": {"coords": (41.9028, 12.4964), "neighbors": ["Milan"]},
    "Warsaw": {"coords": (52.2297, 21.0122), "neighbors": ["Berlin"]},
    # Add more for realism
}

# --- Generalized pathfinding for all modes ---
def find_nearest_waypoint_mode(lat, lon, waypoints):
    min_dist = float('inf')
    nearest = None
    for name, data in waypoints.items():
        d = geodesic((lat, lon), data["coords"]).km
        if d < min_dist:
            min_dist = d
            nearest = name
    return nearest

def find_route_mode(start_name, end_name, waypoints):
    queue = [(0, start_name, [start_name])]
    visited = set()
    while queue:
        cost, current, path = heapq.heappop(queue)
        if current == end_name:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor in waypoints[current]["neighbors"]:
            if neighbor not in visited:
                n_coords = waypoints[neighbor]["coords"]
                c_coords = waypoints[current]["coords"]
                n_cost = cost + geodesic(c_coords, n_coords).km
                heapq.heappush(queue, (n_cost, neighbor, path + [neighbor]))
    return []

# --- Generalized route generation for each mode ---
def generate_mode_path(start, end, waypoints):
    start_wp = find_nearest_waypoint_mode(*start, waypoints)
    end_wp = find_nearest_waypoint_mode(*end, waypoints)
    route_names = find_route_mode(start_wp, end_wp, waypoints)
    path = []
    # If start is not at a waypoint, connect to nearest waypoint
    if waypoints[start_wp]["coords"] != start:
        path.append(start)
        path.append(waypoints[start_wp]["coords"])
    else:
        path.append(start)
    # Add network path
    for name in route_names:
        path.append(waypoints[name]["coords"])
    # If end is not at a waypoint, connect from nearest waypoint
    if waypoints[end_wp]["coords"] != end:
        path.append(end)
    # Fallback: if path is too short, just draw direct line
    if len(path) < 2:
        path = [start, end]
    return path

# --- Update get_road_route and add get_rail_route ---
def get_road_route(self, start: Location, end: Location, mode: str = 'Truck') -> RouteSegment:
    try:
        path = generate_mode_path(start.coordinates, end.coordinates, ROAD_WAYPOINTS)
        distance = sum(geodesic(path[i], path[i+1]).km for i in range(len(path)-1))
        duration = distance / self.mode_params[mode]['avg_speed']
        return RouteSegment(
            mode=mode,
            start=start,
            end=end,
            path=path,
            distance=distance,
            duration=duration,
            co2_emissions=distance * self.mode_params[mode]['co2_per_km'],
            constraints={'vehicle_type': 'electric' if mode == 'Electric Truck' else 'diesel'}
        )
    except Exception as e:
        print(f"Error calculating road route: {e}")
    return None

def get_rail_route(self, start: Location, end: Location) -> RouteSegment:
    try:
        path = generate_mode_path(start.coordinates, end.coordinates, RAIL_WAYPOINTS)
        distance = sum(geodesic(path[i], path[i+1]).km for i in range(len(path)-1))
        duration = distance / self.mode_params['Train']['avg_speed']
        return RouteSegment(
            mode='Train',
            start=start,
            end=end,
            path=path,
            distance=distance,
            duration=duration,
            co2_emissions=distance * self.mode_params['Train']['co2_per_km'],
            constraints={}
        )
    except Exception as e:
        print(f"Error calculating rail route: {e}")
    return None
