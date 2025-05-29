import math
from typing import Tuple, Optional, Dict, Any
import yaml
import random
import os

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing configuration file: {str(e)}")

# Constants
EMISSION_FACTORS = CONFIG['emission_factors']
PACKAGING_EMISSIONS = CONFIG['packaging_emissions']
PACKAGING_COSTS = CONFIG['packaging_costs']
OFFSET_COSTS = CONFIG['offset_costs']
EXCHANGE_RATES = CONFIG['exchange_rates']
LOCATIONS = CONFIG['locations']
CARBON_PRICE_EUR_PER_TON = CONFIG['carbon_price_eur_per_ton']

# Load optimization constants
LOAD_OPTIMIZATION_MIN_CAPACITY = 0.90
LOAD_OPTIMIZATION_MAX_CAPACITY = 0.98
WAREHOUSE_ENERGY_PER_M2 = 100  # kWh per m2
LED_SAVINGS_FACTOR = 0.5
SOLAR_SAVINGS_FACTOR = 0.3
CO2_PER_KWH = 0.5  # kg CO2 per kWh


def calculate_distance(
    country1: str, 
    city1: str, 
    country2: str, 
    city2: str, 
    get_coords_func: Optional[callable] = None
) -> float:
    """
    Calculate great-circle distance using Haversine formula.
    
    Args:
        country1: Source country name
        city1: Source city name
        country2: Destination country name
        city2: Destination city name
        get_coords_func: Optional function to get coordinates (country, city) -> Tuple[float, float]
    
    Returns:
        float: Distance in kilometers
    
    Raises:
        ValueError: If source and destination are the same or if coordinates are not found
    """
    if country1 == country2 and city1 == city2:
        raise ValueError("Source and destination cannot be the same location.")
    
    try:
        if get_coords_func is None:
            coords1 = LOCATIONS.get(country1, {}).get(city1)
            coords2 = LOCATIONS.get(country2, {}).get(city2)
            if coords1 is None:
                raise ValueError(f"Coordinates not found for {city1}, {country1}")
            if coords2 is None:
                raise ValueError(f"Coordinates not found for {city2}, {country2}")
        else:
            coords1 = get_coords_func(country1, city1)
            coords2 = get_coords_func(country2, city2)
            if not isinstance(coords1, tuple) or not isinstance(coords2, tuple):
                raise ValueError("get_coords_func must return a tuple of (latitude, longitude)")
            if len(coords1) != 2 or len(coords2) != 2:
                raise ValueError("Coordinates must be (latitude, longitude) pairs")
    except Exception as e:
        raise ValueError(f"Error getting coordinates: {str(e)}")

    lat1, lon1 = coords1
    lat2, lon2 = coords2
    
    if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")
    
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(R * c, 2)


def calculate_co2(transport_mode: str, distance_km: float, weight_tons: float) -> float:
    """
    Calculate CO2 emissions for a shipment.
    """
    if weight_tons <= 0:
        raise ValueError("Weight must be positive.")
    if distance_km <= 0:
        raise ValueError("Distance must be positive.")
    emission_factor = EMISSION_FACTORS.get(transport_mode)
    if emission_factor is None:
        raise ValueError(f"Invalid transport mode: {transport_mode}")
    co2_kg = distance_km * weight_tons * emission_factor
    return round(co2_kg, 2)


def optimize_route(
    country1: str, city1: str, country2: str, city2: str, distance_km: float, weight_tons: float, prioritize_green: bool = False
) -> Tuple[Tuple[str, float, Optional[str], float], float, Tuple[float, float], Tuple[float, float], float]:
    """
    Optimize transport route to minimize CO2 emissions.
    Returns: (best_option, min_co2, (co2_1, co2_2), (dist1, dist2), current_co2)
    """
    if weight_tons <= 0:
        raise ValueError("Weight must be positive.")
    if distance_km <= 0:
        raise ValueError("Distance must be positive.")
    intercontinental = country1 != country2
    distance_short = distance_km < 1000
    distance_medium = 1000 <= distance_km < 5000
    distance_long = distance_km >= 5000

    current_co2 = distance_km * weight_tons * EMISSION_FACTORS['Truck']
    combinations = []
    if intercontinental:
        if distance_long:
            combinations.extend([
                ('Ship', 0.9, 'Train', 0.1),
                ('Ship', 0.8, 'Electric Truck', 0.2) if prioritize_green else ('Ship', 0.8, 'Truck', 0.2),
                ('Plane', 0.5, 'Ship', 0.5)
            ])
        elif distance_medium:
            combinations.extend([
                ('Ship', 0.7, 'Train', 0.3),
                ('Plane', 0.4, 'Hydrogen Truck', 0.6) if prioritize_green else ('Plane', 0.4, 'Truck', 0.6),
                ('Ship', 0.6, 'Plane', 0.4)
            ])
        else:
            combinations.extend([
                ('Train', 0.8, 'Electric Truck', 0.2) if prioritize_green else ('Train', 0.8, 'Truck', 0.2),
                ('Ship', 0.5, 'Truck', 0.5),
                ('Plane', 0.3, 'Truck', 0.7)
            ])
    else:
        if distance_short:
            combinations.extend([
                ('Train', 0.9, 'Electric Truck', 0.1) if prioritize_green else ('Train', 0.9, 'Truck', 0.1),
                ('Electric Truck', 1.0, None, 0.0) if prioritize_green else ('Truck', 1.0, None, 0.0),
                ('Train', 1.0, None, 0.0)
            ])
        else:
            combinations.extend([
                ('Train', 0.7, 'Biofuel Truck', 0.3) if prioritize_green else ('Train', 0.7, 'Truck', 0.3),
                ('Truck', 0.6, 'Train', 0.4),
                ('Plane', 0.3, 'Truck', 0.7)
            ])

    best_option = None
    min_co2 = float('inf')
    best_breakdown = None
    best_distances = None
    
    for mode1, ratio1, mode2, ratio2 in combinations:
        dist1 = distance_km * ratio1
        dist2 = distance_km * ratio2 if mode2 else 0
        co2_1 = dist1 * weight_tons * EMISSION_FACTORS[mode1]
        co2_2 = dist2 * weight_tons * EMISSION_FACTORS[mode2] if mode2 else 0
        total_co2 = co2_1 + co2_2
        if total_co2 < min_co2:
            min_co2 = total_co2
            best_option = (mode1, ratio1, mode2, ratio2)
            best_breakdown = (co2_1, co2_2)
            best_distances = (dist1, dist2)
    
    return best_option, round(min_co2, 2), best_breakdown, best_distances, round(current_co2, 2)


def calculate_warehouse_savings(
    warehouse_size_m2: float,
    led_percentage: float,
    solar_percentage: float
) -> Tuple[float, float]:
    """
    Calculate CO2 and energy savings from green warehousing technologies.
    
    Args:
        warehouse_size_m2: Size of the warehouse in square meters
        led_percentage: Percentage of LED lighting adoption (0.0 to 1.0)
        solar_percentage: Percentage of solar panel coverage (0.0 to 1.0)
    
    Returns:
        Tuple[float, float]: (CO2 savings in kg, Energy savings in kWh)
    
    Raises:
        ValueError: If warehouse size is not positive or percentages are not between 0 and 1
    """
    if warehouse_size_m2 <= 0:
        raise ValueError("Warehouse size must be positive.")
    if not (0 <= led_percentage <= 1 and 0 <= solar_percentage <= 1):
        raise ValueError("LED and solar percentages must be between 0 and 1 (0% to 100%).")
    
    traditional_energy_kwh = warehouse_size_m2 * WAREHOUSE_ENERGY_PER_M2
    led_savings_kwh = traditional_energy_kwh * led_percentage * LED_SAVINGS_FACTOR
    solar_savings_kwh = traditional_energy_kwh * solar_percentage * SOLAR_SAVINGS_FACTOR
    total_savings_kwh = led_savings_kwh + solar_savings_kwh
    co2_savings_kg = total_savings_kwh * CO2_PER_KWH
    
    return round(co2_savings_kg, 2), round(total_savings_kwh, 2)


def calculate_load_optimization(
    weight_tons: float,
    vehicle_capacity_tons: float,
    avg_trip_distance_km: float = 100
) -> Tuple[int, float]:
    """
    Calculate CO2 savings from efficient load management.
    
    Args:
        weight_tons: Total weight of goods to transport in tons
        vehicle_capacity_tons: Maximum capacity of the vehicle in tons
        avg_trip_distance_km: Average distance per trip in kilometers (default: 100)
    
    Returns:
        Tuple[int, float]: (Number of trips saved, CO2 savings in kg)
    
    Raises:
        ValueError: If weight, capacity, or distance is not positive
    """
    if weight_tons <= 0:
        raise ValueError("Weight must be positive.")
    if vehicle_capacity_tons <= 0:
        raise ValueError("Vehicle capacity must be positive.")
    if avg_trip_distance_km <= 0:
        raise ValueError("Average trip distance must be positive.")
    
    trips_without_optimization = math.ceil(weight_tons / (vehicle_capacity_tons * LOAD_OPTIMIZATION_MIN_CAPACITY))
    optimized_trips = math.ceil(weight_tons / (vehicle_capacity_tons * LOAD_OPTIMIZATION_MAX_CAPACITY))
    trips_saved = max(trips_without_optimization - optimized_trips, 0)
    co2_savings_kg = trips_saved * avg_trip_distance_km * EMISSION_FACTORS['Truck']
    
    return trips_saved, round(co2_savings_kg, 2)


def fetch_carbon_price() -> float:
    """
    Fetch the current carbon price from the configuration with simulated market variation.
    
    This function simulates fetching carbon price from an API (e.g., EU ETS) by adding
    a small random variation to the base price from the configuration.
    
    Returns:
        float: Current carbon price in EUR per ton of CO2
    
    Raises:
        ValueError: If the base carbon price is not found in the configuration
    """
    try:
        base_price = CONFIG.get('carbon_price_eur_per_ton')
        if base_price is None:
            raise ValueError("Base carbon price not found in configuration")
        if not isinstance(base_price, (int, float)) or base_price <= 0:
            raise ValueError("Base carbon price must be a positive number")
            
        # Simulate market variation (±2 EUR)
        variation = random.uniform(-2.0, 2.0)
        current_price = base_price + variation
        
        return round(current_price, 2)
    except Exception as e:
        raise ValueError(f"Error fetching carbon price: {str(e)}")


def calculate_full_logistics_cost(
    payload_kg: float,
    air_distance_km: float,
    sea_distance_km: float,
    rail_distance_km: float,
    scenario: str = 'air',
) -> dict:
    """
    Calculate all logistics and carbon costs for air or multi-modal (sea+rail) shipping.
    Returns a dictionary with all relevant metrics.
    """
    # Load parameters
    ef = CONFIG['emission_factors']
    costs = CONFIG['costs']
    lead_times = CONFIG['lead_times']
    carbon_price = CONFIG['carbon_price_usd_per_ton']
    inventory_value = costs['inventory_value']
    holding_rate = costs['inventory_holding_rate']
    time_cost_per_day = costs['logistics_time_per_day']

    payload_tons = payload_kg / 1000

    if scenario == 'air':
        # Emissions
        co2_tons = payload_tons * air_distance_km * ef['Air'] / 1000
        # Direct cost
        direct_cost = payload_kg * costs['air_freight_per_kg']
        # Lead time
        lead_time_days = lead_times['air_days']
    elif scenario == 'multi-modal':
        # Emissions
        co2_sea = payload_tons * sea_distance_km * ef['Sea'] / 1000
        co2_rail = payload_tons * rail_distance_km * ef['Rail'] / 1000
        co2_tons = co2_sea + co2_rail
        # Direct cost
        direct_cost = costs['sea_freight_per_teu'] + payload_kg * costs['rail_freight_per_kg']
        # Lead time
        lead_time_days = lead_times['multi_modal_days']
    else:
        raise ValueError('Invalid scenario')

    # Logistics time cost
    logistics_time_cost = lead_time_days * time_cost_per_day
    # Inventory holding cost
    inventory_holding_cost = inventory_value * holding_rate * (lead_time_days / 365)
    # Total lead time cost
    total_lead_time_cost = logistics_time_cost + inventory_holding_cost
    # Total financial cost
    total_financial_cost = direct_cost + total_lead_time_cost
    # Carbon cost
    carbon_cost = co2_tons * carbon_price

    return {
        'co2_tons': co2_tons,
        'direct_cost': direct_cost,
        'lead_time_days': lead_time_days,
        'logistics_time_cost': logistics_time_cost,
        'inventory_holding_cost': inventory_holding_cost,
        'total_lead_time_cost': total_lead_time_cost,
        'total_financial_cost': total_financial_cost,
        'carbon_cost': carbon_cost,
    } 
