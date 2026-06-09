import os
import sqlite3
import requests
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import srtm

# Config
DB_PATH = "data/elevation_cache.sqlite"
OUTPUT_DIR = "data"
GRAPH_PATH = os.path.join(OUTPUT_DIR, "berlin_mitte_drive.graphml")
ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
BERLIN_DEFAULT_ELEVATION = 34.0  # Berlin's average elevation in meters

def init_cache_db():
    """Initialize SQLite database for caching elevations."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elevation_cache (
            lat REAL,
            lon REAL,
            elevation REAL,
            PRIMARY KEY (lat, lon)
        )
    """)
    conn.commit()
    conn.close()

def get_cached_elevations(coords):
    """Retrieve elevations from cache for a list of coordinates.
    coords: list of tuples (lat, lon)
    Returns: dict of {(lat, lon): elevation}
    """
    if not coords:
        return {}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We query in chunks to avoid SQLite parameter limits
    chunk_size = 999
    results = {}
    
    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i+chunk_size]
        # Query matching coordinates
        placeholders = ",".join(["(?, ?)"] * len(chunk))
        query = f"SELECT lat, lon, elevation FROM elevation_cache WHERE (lat, lon) IN (VALUES {placeholders})"
        
        # Flatten parameters
        params = []
        for lat, lon in chunk:
            params.extend([round(lat, 6), round(lon, 6)])
            
        cursor.execute(query, params)
        for lat, lon, el in cursor.fetchall():
            results[(lat, lon)] = el
            
    conn.close()
    return results

def save_elevations_to_cache(elevation_data):
    """Save queried elevations to the SQLite cache.
    elevation_data: dict of {(lat, lon): elevation}
    """
    if not elevation_data:
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    insert_data = [
        (round(lat, 6), round(lon, 6), float(el))
        for (lat, lon), el in elevation_data.items()
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO elevation_cache (lat, lon, elevation)
        VALUES (?, ?, ?)
    """, insert_data)
    
    conn.commit()
    conn.close()

def fetch_elevations_from_api(coords_to_fetch):
    """Fetch elevations for coordinates from the Open-Elevation API in batches."""
    if not coords_to_fetch:
        return {}
    
    print(f"Querying Open-Elevation API for {len(coords_to_fetch)} coordinates...")
    results = {}
    batch_size = 150  # Open-Elevation handles smaller batches better
    
    for i in range(0, len(coords_to_fetch), batch_size):
        batch = coords_to_fetch[i:i+batch_size]
        payload = {
            "locations": [{"latitude": lat, "longitude": lon} for lat, lon in batch]
        }
        
        try:
            response = requests.post(ELEVATION_API_URL, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    lat = round(item["latitude"], 6)
                    lon = round(item["longitude"], 6)
                    el = item["elevation"]
                    results[(lat, lon)] = el
            else:
                print(f"API Warning: Status code {response.status_code} received. Using defaults.")
        except Exception as e:
            print(f"API Connection error: {e}. Falling back to default elevations for this batch.")
            # Map default elevation for this batch to avoid failing
            for lat, lon in batch:
                results[(round(lat, 6), round(lon, 6))] = BERLIN_DEFAULT_ELEVATION
                
    return results

def add_elevations_to_graph(G):
    """Query and assign elevation to all nodes in the OSMnx graph using offline SRTM data."""
    nodes = list(G.nodes(data=True))
    coords = [(data['y'], data['x']) for _, data in nodes]
    
    # 1. Check cache
    cached = get_cached_elevations(coords)
    print(f"Found {len(cached)} of {len(coords)} elevations in local SQLite cache.")
    
    # 2. Identify missing coords
    missing_coords = []
    for lat, lon in coords:
        lat_r, lon_r = round(lat, 6), round(lon, 6)
        if (lat_r, lon_r) not in cached:
            missing_coords.append((lat, lon))
            
    # Remove duplicates from missing list
    missing_coords = list(set(missing_coords))
    
    # 3. Query missing coords using srtm
    new_elevations = {}
    if missing_coords:
        print(f"Loading offline SRTM elevation data for {len(missing_coords)} coordinates...")
        try:
            srtm_data = srtm.get_data()
            for lat, lon in missing_coords:
                elevation = srtm_data.get_elevation(lat, lon)
                if elevation is None:
                    elevation = BERLIN_DEFAULT_ELEVATION
                new_elevations[(round(lat, 6), round(lon, 6))] = float(elevation)
            
            # Save new ones to cache
            save_elevations_to_cache(new_elevations)
            print(f"Retrieved and cached {len(new_elevations)} elevations from local SRTM.")
        except Exception as e:
            print(f"SRTM local error: {e}. Falling back to default elevations.")
            for lat, lon in missing_coords:
                new_elevations[(round(lat, 6), round(lon, 6))] = BERLIN_DEFAULT_ELEVATION
        
    # Combine results
    all_elevations = {**cached, **new_elevations}
    
    # 4. Set node attributes in the graph
    node_elevations = {}
    for node_id, data in nodes:
        lat_r, lon_r = round(data['y'], 6), round(data['x'], 6)
        # Fallback to default if somehow missing
        el = all_elevations.get((lat_r, lon_r), BERLIN_DEFAULT_ELEVATION)
        node_elevations[node_id] = el
        
    nx.set_node_attributes(G, values=node_elevations, name="elevation")
    print("Assigned elevation attributes to all nodes.")

def calculate_edge_slopes(G):
    """Compute slope (gradient) for each edge based on node elevations."""
    slopes = {}
    
    for u, v, k, data in G.edges(keys=True, data=True):
        # Retrieve u (source) and v (target) node elevations
        elev_u = G.nodes[u].get("elevation", BERLIN_DEFAULT_ELEVATION)
        elev_v = G.nodes[v].get("elevation", BERLIN_DEFAULT_ELEVATION)
        
        length = data.get("length", 1.0)
        if length <= 0:
            length = 1.0
            
        # Calculate slope as a percentage: (rise / run) * 100
        slope = ((elev_v - elev_u) / length) * 100.0
        
        # Clip extreme values in case of API anomalies
        slope = np.clip(slope, -30.0, 30.0)
        slopes[(u, v, k)] = float(slope)
        
    nx.set_edge_attributes(G, values=slopes, name="slope")
    print("Calculated edge slopes/gradients.")

def download_and_process_berlin_mitte():
    """Main pipeline to download road network, fetch elevation, calculate slopes, and save."""
    print("--- Phase 2: Berlin Map Data Collection & Processing ---")
    init_cache_db()
    
    # Query boundary
    place_name = "Berlin, Germany"
    print(f"Downloading drivable road network for: {place_name}...")
    
    try:
        # Retrieve drive network
        G = ox.graph_from_place(place_name, network_type="drive")
        print(f"Network downloaded successfully! Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        
        # Add elevation features
        add_elevations_to_graph(G)
        
        # Calculate slopes
        calculate_edge_slopes(G)
        
        # Save processed graph
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ox.save_graphml(G, GRAPH_PATH)
        print(f"Saved completed road network graph to {GRAPH_PATH}")
        print("Data collection and initial preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during data collection pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_and_process_berlin_mitte()
