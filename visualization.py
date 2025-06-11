import folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import json
from typing import Dict, List, Tuple, Any
from geopy.distance import geodesic
import math
import logging
import os
from datetime import datetime
import branca.colormap as cm
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys (Placeholders - replace with actual keys in production)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "your_google_maps_api_key")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "your_openweathermap_api_key")
ORS_API_KEY = os.getenv("ORS_API_KEY", "your_openrouteservice_api_key")
MARINETRAFFIC_API_KEY = os.getenv("MARINETRAFFIC_API_KEY", "your_marinetraffic_api_key")
AVIATION_EDGE_API_KEY = os.getenv("AVIATION_EDGE_API_KEY", "your_aviation_edge_api_key")

# Constants for visualization
COLORS = {
    'Electric Truck': '#36A2EB',  # Blue
    'Ship': '#FFCE56',           # Yellow
    'Rail': '#4CAF50',           # Green
    'Flight': '#FF4B4B',         # Red
    'Charging': '#FFA500',       # Orange
    'Refueling': '#800080'       # Purple
}

STYLES = {
    'Electric Truck': {'weight': 4, 'dashArray': 'none'},
    'Ship': {'weight': 4, 'dashArray': '5, 10'},
    'Rail': {'weight': 4, 'dashArray': '10, 10'},
    'Flight': {'weight': 4, 'dashArray': '15, 5'}
}

# Waypoint networks (simplified for demonstration; load from database in production)
ROAD_WAYPOINTS = {
    'USA': [(40.7128, -74.0060), (34.0522, -118.2437)],  # New York, Los Angeles
    'China': [(39.9042, 116.4074), (31.2304, 121.4737)],  # Beijing, Shanghai
    'Germany': [(52.5200, 13.4050), (48.1351, 11.5820)]   # Berlin, Munich
}

RAIL_WAYPOINTS = {
    'USA': [(41.8781, -87.6298), (29.7604, -95.3698)],    # Chicago, Houston
    'China': [(30.2672, 120.1530), (23.1291, 113.2644)],  # Hangzhou, Guangzhou
    'Germany': [(50.9375, 6.9603), (53.5511, 9.9937)]     # Cologne, Hamburg
}

MARITIME_WAYPOINTS = {
    'Ports': [
        {'name': 'Shanghai', 'coords': (31.2304, 121.4737)},
        {'name': 'Rotterdam', 'coords': (51.9225, 4.4792)},
        {'name': 'Los Angeles', 'coords': (33.7683, -118.2161)}
    ]
}

AIRPORT_WAYPOINTS = {
    'USA': [
        {'name': 'JFK', 'coords': (40.6398, -73.7789)},
        {'name': 'LAX', 'coords': (33.9425, -118.4081)}
    ],
    'China': [
        {'name': 'PEK', 'coords': (40.0801, 116.5846)},
        {'name': 'PVG', 'coords': (31.1434, 121.8052)}
    ],
    'Germany': [
        {'name': 'FRA', 'coords': (50.0264, 8.5431)},
        {'name': 'MUC', 'coords': (48.3538, 11.7861)}
    ]
}

def get_weather_conditions(coords: Tuple[float, float]) -> Dict[str, Any]:
    """Fetch real-time weather data for given coordinates."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={coords[0]}&lon={coords[1]}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            'temperature_c': data['main']['temp'],
            'condition': data['weather'][0]['description'],
            'wind_speed_ms': data['wind']['speed'],
            'precipitation': data.get('rain', {}).get('1h', 0)
        }
    except Exception as e:
        logger.error(f"Failed to fetch weather data for {coords}: {e}")
        return {'condition': 'Unknown', 'temperature_c': 0, 'wind_speed_ms': 0, 'precipitation': 0}

def get_maritime_traffic(coords: Tuple[float, float]) -> float:
    """Fetch maritime traffic density (simulated with MarineTraffic API)."""
    try:
        # Placeholder: Replace with actual MarineTraffic API call
        url = f"https://services.marinetraffic.com/api/shipdensity/{MARINETRAFFIC_API_KEY}?lat={coords[0]}&lon={coords[1]}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('density', 1.0)  # Simulated density score
    except Exception as e:
        logger.error(f"Failed to fetch maritime traffic for {coords}: {e}")
        return 1.0

def get_flight_route(source_coords: Tuple[float, float], dest_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Fetch realistic flight path using Aviation Edge API."""
    try:
        # Placeholder: Find nearest airports and get flight path
        url = f"https://aviation-edge.com/v2/public/routes?key={AVIATION_EDGE_API_KEY}&departureLat={source_coords[0]}&departureLon={source_coords[1]}&arrivalLat={dest_coords[0]}&arrivalLon={dest_coords[1]}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data:
            # Simulate great circle path for simplicity
            return generate_great_circle_path(source_coords, dest_coords, steps=50)
        return [source_coords, dest_coords]  # Fallback direct path
    except Exception as e:
        logger.error(f"Failed to fetch flight route: {e}")
        return [source_coords, dest_coords]

def get_road_route(source_coords: Tuple[float, float], dest_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Fetch realistic road route using OpenRouteService API."""
    try:
        headers = {'Authorization': ORS_API_KEY}
        body = {
            'coordinates': [[source_coords[1], source_coords[0]], [dest_coords[1], dest_coords[0]]],
            'profile': 'driving-hgv',  # Heavy goods vehicle
            'geometry': 'true'
        }
        response = requests.post('https://api.openrouteservice.org/v2/directions/driving-hgv/geojson', json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        coords = data['features'][0]['geometry']['coordinates']
        return [(lat, lon) for lon, lat in coords]
    except Exception as e:
        logger.error(f"Failed to fetch road route: {e}")
        return [source_coords, dest_coords]

def get_rail_route(source_coords: Tuple[float, float], dest_coords: Tuple[float, float], waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Generate rail route using waypoints."""
    try:
        path = [source_coords]
        nearest_waypoint = min(waypoints, key=lambda wp: geodesic(source_coords, wp).km, default=source_coords)
        path.append(nearest_waypoint)
        dest_nearest = min(waypoints, key=lambda wp: geodesic(dest_coords, wp).km, default=dest_coords)
        if nearest_waypoint != dest_nearest:
            path.append(dest_nearest)
        path.append(dest_coords)
        return path
    except Exception as e:
        logger.error(f"Failed to generate rail route: {e}")
        return [source_coords, dest_coords]

def get_maritime_route(source_coords: Tuple[float, float], dest_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Generate maritime route with weather and traffic adjustments."""
    try:
        ports = MARITIME_WAYPOINTS['Ports']
        source_port = min(ports, key=lambda p: geodesic(source_coords, p['coords']).km)
        dest_port = min(ports, key=lambda p: geodesic(dest_coords, p['coords']).km)
        path = [source_coords]
        if source_port['coords'] != dest_port['coords']:
            path.append(source_port['coords'])
            # Adjust for maritime traffic and weather
            traffic_density = get_maritime_traffic(source_port['coords'])
            weather = get_weather_conditions(source_port['coords'])
            if traffic_density > 2.0 or weather['wind_speed_ms'] > 15:
                # Simulate detour
                mid_lat = (source_port['coords'][0] + dest_port['coords'][0]) / 2
                mid_lon = (source_port['coords'][1] + dest_port['coords'][1]) / 2 + 5
                path.append((mid_lat, mid_lon))
            path.append(dest_port['coords'])
        path.append(dest_coords)
        return path
    except Exception as e:
        logger.error(f"Failed to generate maritime route: {e}")
        return [source_coords, dest_coords]

def generate_great_circle_path(start: Tuple[float, float], end: Tuple[float, float], steps: int = 50) -> List[Tuple[float, float]]:
    """Generate points along a great circle path."""
    try:
        coords = []
        start_lat, start_lon = np.radians(start)
        end_lat, end_lon = np.radians(end)
        d = 2 * np.arcsin(np.sqrt(
            np.sin((end_lat - start_lat) / 2) ** 2 +
            np.cos(start_lat) * np.cos(end_lat) * np.sin((end_lon - start_lon) / 2) ** 2
        ))
        for i in range(steps + 1):
            f = i / steps
            A = np.sin((1 - f) * d) / np.sin(d)
            B = np.sin(f * d) / np.sin(d)
            x = A * np.cos(start_lat) * np.cos(start_lon) + B * np.cos(end_lat) * np.cos(end_lon)
            y = A * np.cos(start_lat) * np.sin(start_lon) + B * np.cos(end_lat) * np.sin(end_lon)
            z = A * np.sin(start_lat) + B * np.sin(end_lat)
            lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
            lon = np.degrees(np.arctan2(y, x))
            coords.append((lat, lon))
        return coords
    except Exception as e:
        logger.error(f"Failed to generate great circle path: {e}")
        return [start, end]

def generate_mode_path(mode: str, source_coords: Tuple[float, float], dest_coords: Tuple[float, float], country: str) -> List[Tuple[float, float]]:
    """Generate path for a specific transport mode."""
    try:
        if mode == 'Electric Truck':
            return get_road_route(source_coords, dest_coords)
        elif mode == 'Rail':
            waypoints = RAIL_WAYPOINTS.get(country, [])
            return get_rail_route(source_coords, dest_coords, waypoints)
        elif mode == 'Ship':
            return get_maritime_route(source_coords, dest_coords)
        elif mode == 'Flight':
            return get_flight_route(source_coords, dest_coords)
        else:
            return [source_coords, dest_coords]
    except Exception as e:
        logger.error(f"Failed to generate path for mode {mode}: {e}")
        return [source_coords, dest_coords]

def find_nearest_waypoint_mode(mode: str, coords: Tuple[float, float], country: str) -> Tuple[float, float]:
    """Find nearest waypoint for a given mode."""
    try:
        if mode == 'Electric Truck':
            waypoints = ROAD_WAYPOINTS.get(country, [])
        elif mode == 'Rail':
            waypoints = RAIL_WAYPOINTS.get(country, [])
        elif mode == 'Ship':
            waypoints = [p['coords'] for p in MARITIME_WAYPOINTS['Ports']]
        elif mode == 'Flight':
            waypoints = [a['coords'] for a in AIRPORT_WAYPOINTS.get(country, [])]
        else:
            return coords
        return min(waypoints, key=lambda wp: geodesic(coords, wp).km, default=coords)
    except Exception as e:
        logger.error(f"Failed to find nearest waypoint for mode {mode}: {e}")
        return coords

def get_continent(country: str) -> str:
    """Determine continent for a country (simplified mapping)."""
    CONTINENT_MAPPING = {
        'USA': 'North America',
        'China': 'Asia',
        'Germany': 'Europe',
        # Add more mappings as needed
    }
    return CONTINENT_MAPPING.get(country, 'Unknown')

def plan_multi_modal_route(source: Dict, dest: Dict, segments: List[Dict], get_coords_func) -> List[Dict]:
    """Plan a multi-modal route with realistic paths and stops."""
    try:
        legs = []
        current_coords = get_coords_func(source['country'], source['city'])
        source_continent = get_continent(source['country'])
        dest_continent = get_continent(dest['country'])

        for segment in segments:
            mode = segment['mode']
            ratio = segment['ratio']
            distance = segment['distance']
            if ratio <= 0:
                continue

            # Determine segment destination
            if segment == segments[-1]:
                segment_dest_coords = get_coords_func(dest['country'], dest['city'])
            else:
                # Find intermediate hub (e.g., port, airport)
                hub_coords = find_nearest_waypoint_mode(mode, current_coords, source['country'])
                segment_dest_coords = hub_coords

            # Generate realistic path
            path = generate_mode_path(mode, current_coords, segment_dest_coords, source['country'])

            # Add charging/refueling stops for Electric Truck and Flight
            stops = []
            if mode == 'Electric Truck' and distance > 500:  # Assume 500 km range
                num_stops = int(distance // 500)
                for i in range(1, num_stops + 1):
                    stop_idx = int(len(path) * (i / (num_stops + 1)))
                    stop_coords = path[stop_idx]
                    stops.append({
                        'type': 'Charging',
                        'coords': stop_coords,
                        'weather': get_weather_conditions(stop_coords)
                    })
            elif mode == 'Flight' and distance > 5000:  # Assume refueling for long flights
                stop_idx = len(path) // 2
                stop_coords = path[stop_idx]
                stops.append({
                    'type': 'Refueling',
                    'coords': stop_coords,
                    'weather': get_weather_conditions(stop_coords)
                })

            legs.append({
                'mode': mode,
                'path': path,
                'distance': distance,
                'co2': segment['co2'],
                'cost': segment['cost'],
                'stops': stops
            })
            current_coords = segment_dest_coords

        # Ensure final leg reaches destination if needed
        if current_coords != get_coords_func(dest['country'], dest['city']):
            final_path = generate_mode_path('Electric Truck', current_coords, get_coords_func(dest['country'], dest['city']), dest['country'])
            legs.append({
                'mode': 'Electric Truck',
                'path': final_path,
                'distance': geodesic(current_coords, get_coords_func(dest['country'], dest['city'])).km,
                'co2': 0,  # Assume minimal for final short leg
                'cost': 0,
                'stops': []
            })

        return legs
    except Exception as e:
        logger.error(f"Failed to plan multi-modal route: {e}")
        return []

@st.cache_data(ttl=3600)
def render_emission_map(df: pd.DataFrame, get_coords_func, max_routes: int = None) -> folium.Map:
    """Render an interactive emission map with route lines."""
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for emission map.")
            return folium.Map(location=[0, 0], zoom_start=2)

        # Initialize map centered on average coordinates
        coords = []
        for _, row in df.iterrows():
            try:
                source_coords = get_coords_func(row['source_country'], row['source_city'])
                dest_coords = get_coords_func(row['dest_country'], row['dest_city'])
                coords.extend([source_coords, dest_coords])
            except ValueError:
                continue
        if not coords:
            return folium.Map(location=[0, 0], zoom_start=2)
        avg_lat = np.mean([c[0] for c in coords])
        avg_lon = np.mean([c[1] for c in coords])
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles='CartoDB positron')

        # Create colormap for CO2 emissions
        co2_values = df['co2_kg'].values
        colormap = cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=min(co2_values) if len(co2_values) > 0 else 0,
            vmax=max(co2_values) if len(co2_values) > 0 else 1
        )
        colormap.caption = 'CO2 Emissions (kg)'
        m.add_child(colormap)

        # Plot routes
        routes_plotted = 0
        for _, row in df.iterrows():
            if max_routes and routes_plotted >= max_routes:
                break
            try:
                source_coords = get_coords_func(row['source_country'], row['source_city'])
                dest_coords = get_coords_func(row['dest_country'], row['dest_city'])
                co2_kg = row['co2_kg']
                source = row['source']
                destination = row['destination']

                # Draw great circle path
                path = generate_great_circle_path(source_coords, dest_coords)
                folium.PolyLine(
                    locations=path,
                    color=colormap(co2_kg),
                    weight=2,
                    opacity=0.7,
                    popup=f"{source} to {destination}<br>CO2: {co2_kg:.2f} kg"
                ).add_to(m)

                # Add markers
                folium.CircleMarker(
                    location=source_coords,
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"{source}<br>CO2: {co2_kg:.2f} kg"
                ).add_to(m)
                folium.CircleMarker(
                    location=dest_coords,
                    radius=5,
                    color='red',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"{destination}<br>CO2: {co2_kg:.2f} kg"
                ).add_to(m)

                routes_plotted += 1
            except ValueError as e:
                logger.warning(f"Skipping route {source} to {destination}: {e}")
                continue

        return m
    except Exception as e:
        logger.error(f"Failed to render emission map: {e}")
        return folium.Map(location=[0, 0], zoom_start=2)

def render_multi_modal_route(route_data: Dict, get_coords_func) -> folium.Map:
    """Render an interactive multi-modal route map with stops and real-time conditions."""
    try:
        source = route_data['source']
        destination = route_data['destination']
        segments = route_data['segments']
        total_co2 = route_data['total_co2']
        total_cost = route_data['total_cost']
        delivery_time = route_data['delivery_time']

        # Get source and destination coordinates
        source_coords = get_coords_func(source['country'], source['city'])
        dest_coords = get_coords_func(destination['country'], destination['city'])

        # Initialize map
        avg_lat = (source_coords[0] + dest_coords[0]) / 2
        avg_lon = (source_coords[1] + dest_coords[1]) / 2
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles='CartoDB positron')

        # Plan multi-modal route
        legs = plan_multi_modal_route(source, destination, segments, get_coords_func)

        # Draw each leg
        for leg in legs:
            mode = leg['mode']
            path = leg['path']
            distance = leg['distance']
            co2 = leg['co2']
            cost = leg['cost']
            stops = leg['stops']

            # Draw path
            folium.PolyLine(
                locations=path,
                color=COLORS.get(mode, '#000000'),
                weight=STYLES.get(mode, {}).get('weight', 4),
                dash_array=STYLES.get(mode, {}).get('dashArray', 'none'),
                popup=f"{mode}<br>Distance: {distance:.2f} km<br>CO2: {co2:.2f} kg<br>Cost: ${cost:.2f}"
            ).add_to(m)

            # Add stops (charging/refueling)
            for stop in stops:
                stop_type = stop['type']
                stop_coords = stop['coords']
                weather = stop['weather']
                folium.Marker(
                    location=stop_coords,
                    icon=folium.Icon(color='orange' if stop_type == 'Charging' else 'purple', icon='bolt'),
                    popup=f"{stop_type} Stop<br>Weather: {weather['condition']}<br>Temp: {weather['temperature_c']}°C"
                ).add_to(m)

        # Add source and destination markers
        folium.Marker(
            location=source_coords,
            icon=folium.Icon(color='blue', icon='play'),
            popup=f"Start: {source['city']}, {source['country']}<br>Total CO2: {total_co2:.2f} kg<br>Total Cost: ${total_cost:.2f}"
        ).add_to(m)
        folium.Marker(
            location=dest_coords,
            icon=folium.Icon(color='red', icon='stop'),
            popup=f"End: {destination['city']}, {destination['country']}<br>Delivery Time: {delivery_time:.2f} days"
        ).add_to(m)

        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey;">
            <h4>Transport Modes</h4>
        """
        for mode, color in COLORS.items():
            if mode in ['Charging', 'Refueling']:
                continue
            legend_html += f'<p><span style="color:{color};">&#9473;</span> {mode}</p>'
        legend_html += """
            <p><span style="color:orange;">&#9889;</span> Charging Stop</p>
            <p><span style="color:purple;">&#9889;</span> Refueling Stop</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        return m
    except Exception as e:
        logger.error(f"Failed to render multi-modal route: {e}")
        return folium.Map(location=[0, 0], zoom_start=2)

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Create a bar chart using Plotly."""
    try:
        fig = px.bar(df, x=x, y=y, title=title)
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            template='plotly_white'
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create bar chart: {e}")
        return go.Figure()

def pie_chart(df: pd.DataFrame, values: str, names: str, title: str) -> go.Figure:
    """Create a pie chart using Plotly."""
    try:
        fig = px.pie(df, values=values, names=names, title=title)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        logger.error(f"Failed to create pie chart: {e}")
        return go.Figure()

def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Create a line chart using Plotly."""
    try:
        fig = px.line(df, x=x, y=y, title=title)
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            template='plotly_white'
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create line chart: {e}")
        return go.Figure()
