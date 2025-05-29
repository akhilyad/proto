import sqlite3
import logging
from typing import Optional, Tuple, List, Dict, Any, ContextManager
import pandas as pd
import uuid
import datetime
import yaml
import os
from contextlib import contextmanager
from pathlib import Path
import threading
from queue import Queue, Empty, Full
import time
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing configuration file: {str(e)}")

# Database configuration
DB_DIR = os.getenv('DB_DIR', os.path.dirname(__file__))
DB_PATH = os.path.join(DB_DIR, 'emissions.db')
MAX_CONNECTIONS = int(os.getenv('MAX_DB_CONNECTIONS', '10'))
CONNECTION_TIMEOUT = int(os.getenv('DB_CONNECTION_TIMEOUT', '30'))

# Connection pool
_connection_pool = Queue(maxsize=MAX_CONNECTIONS)
_pool_lock = threading.Lock()

def _init_connection_pool():
    """Initialize the connection pool with MAX_CONNECTIONS connections."""
    with _pool_lock:
        while not _connection_pool.empty():
            try:
                conn = _connection_pool.get_nowait()
                conn.close()
            except Empty:
                break
            except Exception as e:
                logging.error(f"Error closing existing connection: {e}")
    
    for _ in range(MAX_CONNECTIONS):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=CONNECTION_TIMEOUT)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=-2000")  # Use 2MB cache
            _connection_pool.put(conn)
        except sqlite3.Error as e:
            logging.error(f"Failed to create database connection: {e}")
            raise

def _get_connection() -> sqlite3.Connection:
    """Get a connection from the pool with timeout."""
    try:
        return _connection_pool.get(timeout=CONNECTION_TIMEOUT)
    except Empty:
        raise sqlite3.Error("No database connections available")

def _return_connection(conn: sqlite3.Connection):
    """Return a connection to the pool."""
    try:
        _connection_pool.put(conn, timeout=1)
    except Queue.Full:
        conn.close()  # Close the connection if pool is full

@contextmanager
def get_db_connection() -> ContextManager[sqlite3.Connection]:
    """Context manager for database connections using connection pool."""
    conn = None
    try:
        conn = _get_connection()
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        raise
    except Exception as e:
        logging.error(f"Unexpected error in database connection: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        raise
    finally:
        if conn:
            try:
                if not _connection_pool.full():
                    _return_connection(conn)
                else:
                    conn.close()
            except Exception as e:
                logging.error(f"Error returning connection to pool: {e}")
                try:
                    conn.close()
                except Exception:
                    pass

def insert_sample_suppliers():
    """Insert sample suppliers if the suppliers table is empty."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM suppliers')
            count = c.fetchone()[0]
            if count == 0:
                suppliers = [
                    (str(uuid.uuid4()), 'EcoSteel Ltd', 'Germany', 'Berlin', 'Steel', 85, 10000, 'Recycled materials, renewable energy'),
                    (str(uuid.uuid4()), 'GreenPlastics', 'USA', 'Chicago', 'Plastic', 70, 5000, 'Biodegradable plastics'),
                    (str(uuid.uuid4()), 'BioPack', 'France', 'Paris', 'Biodegradable', 90, 2000, 'Compostable packaging'),
                    (str(uuid.uuid4()), 'RenewMetals', 'UK', 'London', 'Metals', 80, 8000, 'Low-emission smelting'),
                ]
                c.executemany('''
                    INSERT INTO suppliers (id, supplier_name, country, city, material, green_score, annual_capacity_tons, sustainable_practices)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', suppliers)
                conn.commit()
    except Exception as e:
        logging.error(f"Failed to insert sample suppliers: {e}")

def init_db():
    """Initialize the SQLite database with required tables and sample data."""
    try:
        # Ensure database directory exists
        os.makedirs(DB_DIR, exist_ok=True)
        
        # Initialize connection pool
        _init_connection_pool()
        
        with get_db_connection() as conn:
            c = conn.cursor()
            # Create tables with proper constraints and indexes
            c.executescript('''
                CREATE TABLE IF NOT EXISTS emissions (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    transport_mode TEXT NOT NULL,
                    distance_km REAL NOT NULL CHECK (distance_km > 0),
                    co2_kg REAL NOT NULL CHECK (co2_kg >= 0),
                    weight_tons REAL NOT NULL CHECK (weight_tons > 0),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_emissions_timestamp ON emissions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_emissions_source ON emissions(source);
                CREATE INDEX IF NOT EXISTS idx_emissions_destination ON emissions(destination);
                
                CREATE TABLE IF NOT EXISTS packaging (
                    id TEXT PRIMARY KEY,
                    material_type TEXT NOT NULL,
                    weight_kg REAL NOT NULL CHECK (weight_kg > 0),
                    co2_kg REAL NOT NULL CHECK (co2_kg >= 0),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS offsets (
                    id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    co2_kg REAL NOT NULL CHECK (co2_kg > 0),
                    cost_eur REAL NOT NULL CHECK (cost_eur >= 0),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS suppliers (
                    id TEXT PRIMARY KEY,
                    supplier_name TEXT NOT NULL,
                    country TEXT NOT NULL,
                    city TEXT NOT NULL,
                    material TEXT NOT NULL,
                    green_score INTEGER NOT NULL,
                    annual_capacity_tons REAL NOT NULL,
                    sustainable_practices TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TRIGGER IF NOT EXISTS update_emissions_timestamp 
                AFTER UPDATE ON emissions
                BEGIN
                    UPDATE emissions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
                
                CREATE TRIGGER IF NOT EXISTS update_packaging_timestamp 
                AFTER UPDATE ON packaging
                BEGIN
                    UPDATE packaging SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
                
                CREATE TRIGGER IF NOT EXISTS update_offsets_timestamp 
                AFTER UPDATE ON offsets
                BEGIN
                    UPDATE offsets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
            ''')
            conn.commit()
            logging.info("Database initialized successfully")
        # Insert sample suppliers if needed
        insert_sample_suppliers()
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize database: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error initializing database: {e}")
        raise

def cleanup_old_records(retention_days: int = 365) -> None:
    """Remove records older than retention_days from emissions, packaging, and offsets tables."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
            c.execute('DELETE FROM emissions WHERE timestamp < ?', (cutoff_date,))
            c.execute('DELETE FROM packaging WHERE timestamp < ?', (cutoff_date,))
            c.execute('DELETE FROM offsets WHERE timestamp < ?', (cutoff_date,))
            conn.commit()
            logging.info(f"Cleaned up records older than {cutoff_date}")
    except sqlite3.Error as e:
        logging.error(f"Database cleanup failed: {e}")

def save_emission(source: str, destination: str, transport_mode: str, distance_km: float, co2_kg: float, weight_tons: float) -> None:
    """Save emission data with proper validation."""
    if not all([source, destination, transport_mode]):
        raise ValueError("All fields must be non-empty")
    if not all(isinstance(x, (int, float)) and x > 0 for x in [distance_km, co2_kg, weight_tons]):
        raise ValueError("Distance, CO2, and weight must be positive numbers")
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            emission_id = str(uuid.uuid4())
            c.execute('''
                INSERT INTO emissions (id, source, destination, transport_mode, distance_km, co2_kg, weight_tons)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (emission_id, source, destination, transport_mode, distance_km, co2_kg, weight_tons))
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Failed to save emission: {e}")
        raise

def save_packaging(material_type: str, weight_kg: float, co2_kg: float) -> None:
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            packaging_id = str(uuid.uuid4())
            c.execute('INSERT INTO packaging (id, material_type, weight_kg, co2_kg) VALUES (?, ?, ?, ?)',
                      (packaging_id, material_type, weight_kg, co2_kg))
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Failed to save packaging: {e}")

def save_offset(project_type: str, co2_offset_tons: float, cost_usd: float) -> None:
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            offset_id = str(uuid.uuid4())
            c.execute('INSERT INTO offsets (id, project_type, co2_offset_tons, cost_usd) VALUES (?, ?, ?, ?)',
                      (offset_id, project_type, co2_offset_tons, cost_usd))
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Failed to save offset: {e}")

def get_emissions() -> pd.DataFrame:
    """Get emissions data with proper error handling and type conversion."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query('SELECT * FROM emissions', conn)
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except sqlite3.Error as e:
        logging.error(f"Failed to retrieve emissions: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error retrieving emissions: {e}")
        return pd.DataFrame()

def get_packaging() -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query('SELECT * FROM packaging', conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            return df
    except sqlite3.Error as e:
        logging.error(f"Failed to retrieve packaging: {e}")
        return pd.DataFrame()

def get_offsets() -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query('SELECT * FROM offsets', conn)
        return df
    except sqlite3.Error as e:
        logging.error(f"Failed to retrieve offsets: {e}")
        return pd.DataFrame()

def get_suppliers(country: Optional[str] = None, city: Optional[str] = None, material: Optional[str] = None, min_green_score: int = 0, min_date: Optional[str] = None) -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            query = 'SELECT * FROM suppliers WHERE green_score >= ?'
            params = [min_green_score]
            conditions = []
            if country and country != "All":
                conditions.append('country = ?')
                params.append(country)
            if city and city != "All":
                conditions.append('city = ?')
                params.append(city)
            if material:
                conditions.append('LOWER(material) LIKE ?')
                params.append(f'%{material.lower()}%')
            if min_date:
                conditions.append('created_at >= ?')
                params.append(min_date)
            if conditions:
                query += ' AND ' + ' AND '.join(conditions)
            df = pd.read_sql_query(query, conn, params=params)
        return df
    except sqlite3.Error as e:
        logging.error(f"Failed to retrieve suppliers: {e}")
        return pd.DataFrame()

def _get_coordinates_from_config(country: str, city: str):
    """
    Returns (latitude, longitude) for the given country and city from CONFIG.
    Returns None if not found.
    """
    try:
        loc = CONFIG['locations'][country][city]
        return float(loc['lat']), float(loc['lon'])
    except KeyError:
        return None

def geocode_location(city, country):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"city": city, "country": country, "format": "json"}
    response = requests.get(url, params=params, headers={"User-Agent": "C360-App"})
    if response.ok and response.json():
        data = response.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None

def get_coordinates(country, city):
    coords = _get_coordinates_from_config(country, city)
    if coords:
        return coords
    # Fallback to geocoding
    return geocode_location(city, country)
