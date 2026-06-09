import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

# Config
GRAPH_PATH = "data/berlin_mitte_drive.graphml"
DRIVERS_OUTPUT = "data/driver_profiles.csv"
LOGS_OUTPUT = "data/simulated_driving_logs.csv"

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_driver_profiles(n_drivers=200):
    """Generate a population of synthetic driver profiles."""
    profiles = []
    
    vehicle_types = ['Compact', 'Sedan', 'SUV', 'RV']
    vehicle_weights = [0.25, 0.45, 0.20, 0.10]
    
    for i in range(n_drivers):
        driver_id = f"driver_{i:03d}"
        age = int(random.randint(18, 85))
        
        # Experience is bounded by age
        max_experience = max(0, age - 18)
        experience_years = float(round(random.uniform(0.0, min(max_experience, 50.0)), 1))
        
        # Vehicle type
        vehicle_class = np.random.choice(vehicle_types, p=vehicle_weights)
        
        # Set vehicle dimensions
        if vehicle_class == 'Compact':
            vehicle_width = 1.7
            vehicle_length = 3.8
            vehicle_weight = 1200 # kg
        elif vehicle_class == 'Sedan':
            vehicle_width = 1.9
            vehicle_length = 4.6
            vehicle_weight = 1600
        elif vehicle_class == 'SUV':
            vehicle_width = 2.1
            vehicle_length = 4.8
            vehicle_weight = 2200
        else: # RV
            vehicle_width = 2.5
            vehicle_length = 7.5
            vehicle_weight = 4500
            
        # Comfort metrics (0.0 to 1.0)
        # Novices (<2 years) and elderly (>70) have lower general comfort
        base_comfort = 0.8
        if experience_years < 2.0:
            base_comfort -= 0.3
        if age > 70:
            base_comfort -= 0.2
            
        highway_comfort = float(np.clip(base_comfort + random.uniform(-0.2, 0.2), 0.1, 1.0))
        narrow_road_comfort = float(np.clip(base_comfort + random.uniform(-0.3, 0.2), 0.1, 1.0))
        
        # Night vision decreases with age
        night_vision = 1.0
        if age > 50:
            night_vision -= 0.01 * (age - 50)
        night_vision = float(np.clip(night_vision + random.uniform(-0.1, 0.1), 0.2, 1.0))
        
        profiles.append({
            "driver_id": driver_id,
            "age": age,
            "experience_years": experience_years,
            "vehicle_class": vehicle_class,
            "vehicle_width": vehicle_width,
            "vehicle_length": vehicle_length,
            "vehicle_weight": vehicle_weight,
            "highway_comfort": round(highway_comfort, 2),
            "narrow_road_comfort": round(narrow_road_comfort, 2),
            "night_vision": round(night_vision, 2)
        })
        
    df = pd.DataFrame(profiles)
    os.makedirs(os.path.dirname(DRIVERS_OUTPUT), exist_ok=True)
    df.to_csv(DRIVERS_OUTPUT, index=False)
    print(f"Generated {n_drivers} driver profiles and saved to {DRIVERS_OUTPUT}")
    return df

def get_traffic_factor(highway_type, hour):
    """Simulate traffic congestion factor (1.0 to 4.0) based on road class and hour."""
    # Base traffic susceptibility by road class
    # Motorways and primary roads are highly prone to traffic. Residential roads are not.
    base_demand = {
        'motorway': 0.85,
        'motorway_link': 0.70,
        'trunk': 0.80,
        'trunk_link': 0.65,
        'primary': 0.75,
        'primary_link': 0.60,
        'secondary': 0.50,
        'secondary_link': 0.45,
        'tertiary': 0.30,
        'tertiary_link': 0.25,
        'residential': 0.10,
        'living_street': 0.05,
        'unclassified': 0.15
    }
    
    # Handle lists (OSMnx sometimes returns list for highway tag)
    if isinstance(highway_type, list):
        highway_type = highway_type[0]
        
    demand = base_demand.get(highway_type, 0.20)
    
    # Diurnal peak curves (morning rush hour, evening rush hour)
    if 7 <= hour <= 9:  # Morning Rush Hour
        diurnal_mult = 3.2
    elif 16 <= hour <= 18:  # Evening Rush Hour
        diurnal_mult = 3.5
    elif 10 <= hour <= 15:  # Middle of the day
        diurnal_mult = 1.3
    elif 19 <= hour <= 21:  # Evening decline
        diurnal_mult = 1.0
    elif 22 <= hour <= 23 or 5 <= hour <= 6:  # Late night / early morning
        diurnal_mult = 0.5
    else:  # Dead of night (0-4)
        diurnal_mult = 0.15
        
    # Congestion factor calculation
    noise = random.uniform(0.8, 1.2)
    congestion = demand * diurnal_mult * noise
    
    # Scale from 1.0 (free flow) to 4.0 (gridlock)
    traffic_factor = 1.0 + 3.0 * np.clip(congestion, 0.0, 1.0)
    return float(round(traffic_factor, 2))

def parse_geometry_sinuosity(data, length):
    """Compute sinuosity of an edge. Ratio of actual length to straight-line distance."""
    if "geometry" in data:
        try:
            # Parse geom string back to coordinates or extract points
            # OSMnx save_graphml stores geometries as string representations
            geom_str = data["geometry"]
            # A simple fallback check: if we can't parse it, we use length-based heuristics.
            # But let's write a simple parser for LineString string representations
            if geom_str.startswith("LINESTRING"):
                # E.g. LINESTRING (13.4040601 52.5178855, 13.4045 52.5180)
                coords_str = geom_str.replace("LINESTRING (", "").replace(")", "")
                coords = [list(map(float, pt.split())) for pt in coords_str.split(", ")]
                if len(coords) >= 2:
                    p1 = coords[0]
                    pn = coords[-1]
                    # Compute straight-line distance in degrees (approximate)
                    # We can use simple Euclidean scale for Berlin coordinates:
                    # 1 deg lat ~ 111km, 1 deg lon ~ 68km at 52.5N
                    dy = (pn[1] - p1[1]) * 111000.0
                    dx = (pn[0] - p1[0]) * 68000.0
                    straight_length = np.sqrt(dx**2 + dy**2)
                    if straight_length > 0:
                        sinuosity = length / straight_length
                        # Cap it to reasonable numbers
                        return float(np.clip(sinuosity, 1.0, 3.5))
        except Exception:
            pass
    return 1.0

def calculate_subjective_difficulty(driver, edge_data, hour, traffic_factor):
    """Compute personalized driving difficulty score (1.0 to 5.0)."""
    # 1. Base difficulty
    difficulty = 1.0
    
    # 2. Extract road tags safely
    road_class = edge_data.get("highway", "residential")
    if isinstance(road_class, list):
        road_class = road_class[0]
        
    length = float(edge_data.get("length", 10.0))
    slope = float(edge_data.get("slope", 0.0))
    is_one_way = edge_data.get("oneway", "False") == "True"
    surface = edge_data.get("surface", "asphalt")
    is_lit = edge_data.get("lit", "yes") == "yes"
    has_tram = "railway" in edge_data or edge_data.get("railway") == "tram"
    
    # Estimate lanes
    lanes_val = edge_data.get("lanes", "1")
    if isinstance(lanes_val, list):
        lanes_val = lanes_val[0]
    try:
        lane_count = int(float(lanes_val))
    except ValueError:
        lane_count = 1
        
    # Estimate width
    width_val = edge_data.get("width", None)
    if isinstance(width_val, list):
        width_val = width_val[0]
    try:
        road_width = float(width_val) if width_val else None
    except ValueError:
        road_width = None
        
    if not road_width:
        # Assign defaults based on classification
        width_defaults = {
            "motorway": 12.0, "primary": 8.5, "secondary": 7.0,
            "tertiary": 6.0, "residential": 4.5, "living_street": 3.5
        }
        road_width = width_defaults.get(road_class, 5.0)
        
    # Compute sinuosity (windingness)
    sinuosity = parse_geometry_sinuosity(edge_data, length)
    
    # --- Apply Personalization Rules ---
    
    # Rule A: Narrow Road Stress (Width vs. Vehicle Width)
    margin = road_width - driver["vehicle_width"]
    if margin < 0.5:
        difficulty += 2.0  # Extremely tight fit
    elif margin < 1.0:
        difficulty += 1.0  # Narrow fit
        
    # Narrow Two-Way street stress (Passing oncoming traffic)
    if not is_one_way and road_width < 5.0:
        if driver["vehicle_class"] in ["SUV", "RV"]:
            difficulty += 1.5  # High anxiety passing cars
            
    # Rule B: Winding Curve Stress
    if sinuosity > 1.2:
        curve_stress = (sinuosity - 1.0) * 2.0 * (1.0 - driver["narrow_road_comfort"])
        difficulty += curve_stress
        if driver["experience_years"] < 2.0:
            difficulty += 0.8  # Novices struggle on sharp curves
            
    # Rule C: Slope stress for large vehicles
    if abs(slope) > 5.0:
        if driver["vehicle_class"] == "RV":
            difficulty += 1.2  # RVs struggle on steep gradients
        if driver["experience_years"] < 2.0:
            difficulty += 0.4
            
    # Rule D: Night Visibility Stress
    is_night = (hour >= 20 or hour <= 6)
    if is_night:
        if not is_lit:
            # High stress in unlit forest/neighborhood roads
            night_stress = 2.0 * (1.0 - driver["night_vision"])
            difficulty += night_stress
            if driver["age"] > 65:
                difficulty += 1.0  # Older drivers struggle with night glare/contrast
        else:
            # Even on lit roads, night driving adds minor stress for elders
            if driver["age"] > 70:
                difficulty += 0.5
                
    # Rule E: Heavy Traffic & Lane Merging Stress
    if traffic_factor > 2.2:
        traffic_stress = (traffic_factor - 1.0) * 0.5
        if driver["experience_years"] < 2.0:
            traffic_stress *= 1.8  # Novices get highly stressed in traffic
            difficulty += 1.0
        difficulty += traffic_stress
        
        # High lane count merges under heavy traffic
        if lane_count >= 3:
            difficulty += 0.8
            
    # Rule F: Tram Track stress
    if has_tram and driver["experience_years"] < 3.0:
        difficulty += 1.0  # Tram track tire-slip anxiety
        
    # Rule G: Surface Quality
    if surface == "cobblestone":
        if driver["vehicle_class"] == "Compact":
            difficulty += 0.8  # Bumpy and uncomfortable
        elif driver["vehicle_class"] == "RV":
            difficulty += 0.5
            
    # 3. Add dynamic context (e.g. driving in forest/jungle)
    # Check if road is inside a forest region (simulate using a simple bounding box or tag)
    # Berlin Mitte has park areas like the Tiergarten. We simulate park roads.
    name_val = edge_data.get("name", "")
    if isinstance(name_val, list):
        name_val = " ".join([str(n) for n in name_val])
    name_lower = str(name_val).lower()
    is_park_forest = ("tiergarten" in name_lower or road_class == "living_street")
    if is_park_forest:
        difficulty += 0.3
        if is_night and not is_lit:
            difficulty += 1.0  # Extra dark, animal hazards
            
    # 4. Add slight random noise (simulate mood/events)
    difficulty += random.normalvariate(0.0, 0.15)
    
    # 5. Clip difficulty score to range [1.0, 5.0]
    return float(np.clip(difficulty, 1.0, 5.0))

def simulate_trips(G, drivers_df, n_trips_per_driver=50):
    """Simulate a set of trips for each driver, logging edge-level experiences."""
    print(f"Loading Berlin Mitte network graph for trip simulation...")
    
    # Get largest strongly connected component to guarantee paths
    scc = max(nx.strongly_connected_components(G), key=len)
    G_sub = G.subgraph(scc).copy()
    print(f"Filtered to largest strongly connected component: {len(G_sub.nodes)} nodes.")
    
    all_logs = []
    drivers_list = drivers_df.to_dict(orient="records")
    
    nodes_list = list(G_sub.nodes)
    total_trips = len(drivers_list) * n_trips_per_driver
    print(f"Simulating {total_trips} trips in total ({n_trips_per_driver} per driver)...")
    
    trip_count = 0
    for driver in drivers_list:
        for _ in range(n_trips_per_driver):
            # Select random source and target
            source = random.choice(nodes_list)
            target = random.choice(nodes_list)
            while source == target:
                target = random.choice(nodes_list)
                
            # Random hour of day
            hour = random.randint(0, 23)
            
            try:
                # Find shortest path based on length
                path = nx.shortest_path(G_sub, source=source, target=target, weight="length")
                
                # For each edge in the path, simulate driver experience
                for idx in range(len(path) - 1):
                    u = path[idx]
                    v = path[idx + 1]
                    
                    # Graph holds multiple edges between same nodes in MultiDiGraph
                    # Get the edge details (typically key 0)
                    edge_dict = G_sub.get_edge_data(u, v)
                    if not edge_dict:
                        continue
                    
                    key = list(edge_dict.keys())[0]
                    data = edge_dict[key]
                    
                    road_class = data.get("highway", "residential")
                    if isinstance(road_class, list):
                        road_class = road_class[0]
                        
                    # Calculate traffic
                    traffic_factor = get_traffic_factor(road_class, hour)
                    
                    # Calculate subjective difficulty
                    difficulty = calculate_subjective_difficulty(driver, data, hour, traffic_factor)
                    
                    # Calculate simulated actual speed
                    maxspeed_val = data.get("maxspeed", "50")
                    if isinstance(maxspeed_val, list):
                        maxspeed_val = maxspeed_val[0]
                    try:
                        speed_limit = float(maxspeed_val.replace("DE:urban", "50").replace("DE:zone:30", "30"))
                    except ValueError:
                        speed_limit = 50.0
                        
                    # Speed reduction based on traffic and driver caution
                    caution_factor = 1.0 - (driver["experience_years"] / 100.0) # novices drive slightly slower
                    speed_mult = (1.0 / traffic_factor) * random.uniform(0.9, 1.1) * caution_factor
                    actual_speed = float(round(np.clip(speed_limit * speed_mult, 5.0, speed_limit), 1))
                    
                    # Extract surface/lit/tram
                    surface = data.get("surface", "asphalt")
                    if isinstance(surface, list):
                        surface = surface[0]
                        
                    is_lit = 1 if data.get("lit", "yes") == "yes" else 0
                    has_tram = 1 if ("railway" in data or data.get("railway") == "tram") else 0
                    
                    # Build log entry
                    all_logs.append({
                        # Driver attributes
                        "driver_id": driver["driver_id"],
                        "driver_age": driver["age"],
                        "driver_experience": driver["experience_years"],
                        "vehicle_class": driver["vehicle_class"],
                        "vehicle_width": driver["vehicle_width"],
                        "vehicle_weight": driver["vehicle_weight"],
                        "highway_comfort": driver["highway_comfort"],
                        "narrow_road_comfort": driver["narrow_road_comfort"],
                        "night_vision": driver["night_vision"],
                        # Trip attributes
                        "hour": hour,
                        "is_night": 1 if (hour >= 20 or hour <= 6) else 0,
                        # Road features (15 mapped features)
                        "edge_u": u,
                        "edge_v": v,
                        "edge_key": key,
                        "length": float(data.get("length", 10.0)),
                        "slope": float(data.get("slope", 0.0)),
                        "road_class": road_class,
                        "surface": surface,
                        "is_lit": is_lit,
                        "has_tram": has_tram,
                        "is_one_way": 1 if data.get("oneway", "False") == "True" else 0,
                        "lanes": int(data.get("lanes", 1) if isinstance(data.get("lanes", 1), int) else 1),
                        "sinuosity": parse_geometry_sinuosity(data, float(data.get("length", 10.0))),
                        # Dynamic environment features
                        "traffic_factor": traffic_factor,
                        "actual_speed": actual_speed,
                        # Target variable
                        "difficulty_score": difficulty
                    })
            except nx.NetworkXNoPath:
                continue
                
            trip_count += 1
            if trip_count % 1000 == 0:
                print(f"Simulated {trip_count} / {total_trips} trips...")
                
    # Save to CSV
    logs_df = pd.DataFrame(all_logs)
    logs_df.to_csv(LOGS_OUTPUT, index=False)
    print(f"Successfully simulated {len(logs_df)} segment-level driving logs.")
    print(f"Logs dataset saved to {LOGS_OUTPUT}")
    return logs_df

def run_simulation_pipeline():
    """Main execution block for Phase 3."""
    print("--- Phase 3: Traffic & Driver Behavior Simulation ---")
    
    # Check if graph exists
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Missing road graph at {GRAPH_PATH}. Please run downloader.py first.")
        
    G = ox.load_graphml(GRAPH_PATH)
    
    # 1. Generate driver profiles
    drivers_df = generate_driver_profiles(n_drivers=200)
    
    # 2. Simulate paths and compute personalized stress ratings
    simulate_trips(G, drivers_df, n_trips_per_driver=40)
    print("Phase 3 pipeline completed successfully!")

if __name__ == "__main__":
    run_simulation_pipeline()
