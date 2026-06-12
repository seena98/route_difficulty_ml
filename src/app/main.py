import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import random

# Config
GRAPH_PATH = "data/berlin_mitte_drive.graphml"
MODEL_PATH_XGB = "data/best_xgb_model.pkl"
MODEL_PATH_RF = "data/best_rf_model.pkl"
FEATURE_LIST_PATH = "data/processed/feature_list.txt"

# Set page configuration with premium styling
st.set_page_config(page_title="Personalized Navigation Systems - Route Difficulty Predictor", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1 {
        color: #1e3d59;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    h3 {
        color: #17b978;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# 1. LANDMARKS FOR CENTRAL ROUTING IN BERLIN
LANDMARKS = {
    "Brandenburg Gate": (52.516275, 13.377704),
    "Alexanderplatz (TV Tower)": (52.520815, 13.409419),
    "Hauptbahnhof (Central Station)": (52.525010, 13.369402),
    "Potsdamer Platz": (52.509647, 13.375944),
    "Checkpoint Charlie": (52.507443, 13.390391),
    "Victory Column (Siegessäule)": (52.514521, 13.350116),
    "Museum Island": (52.518625, 13.399587),
    "Spandauer Vorstadt (Hackescher Markt)": (52.522851, 13.401912),
    "Schloss Charlottenburg (Palace)": (52.521193, 13.295777),
    "Kurfürstendamm (Shopping Avenue)": (52.503611, 13.329444),
    "Tempelhofer Feld (Airport Park)": (52.473022, 13.401811),
    "East Side Gallery (Berlin Wall)": (52.505018, 13.439703),
    "Schloss Bellevue (Presidential Palace)": (52.521389, 13.350278),
    "Kottbusser Tor (Kreuzberg Hub)": (52.499044, 13.418204),
    "Treptower Park (Soviet Memorial)": (52.486221, 13.471602),
    "Funkturm (Messe Berlin / Westend)": (52.504193, 13.279644),
    "Botanischer Garten (Steglitz)": (52.458298, 13.306002),
    "Ostbahnhof (East Station)": (52.510300, 13.434800),
    "Gendarmenmarkt (Historic Square)": (52.513611, 13.392694)
}

@st.cache_resource
def load_resources():
    """Load graph and machine learning models, caching them for fast UI loading."""
    if not os.path.exists(GRAPH_PATH):
        st.error(f"Missing road graph at {GRAPH_PATH}. Run downloader.py first.")
        st.stop()
        
    G = ox.load_graphml(GRAPH_PATH)
    
    # Extract strongly connected component to guarantee paths
    scc = max(nx.strongly_connected_components(G), key=len)
    G_sub = G.subgraph(scc).copy()
    
    # Load Models
    model_xgb = None
    model_rf = None
    if os.path.exists(MODEL_PATH_XGB):
        with open(MODEL_PATH_XGB, "rb") as f:
            model_xgb = pickle.load(f)
    if os.path.exists(MODEL_PATH_RF):
        with open(MODEL_PATH_RF, "rb") as f:
            model_rf = pickle.load(f)
            
    # Load feature column names
    feature_cols = []
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, "r") as f:
            feature_cols = [line.strip() for line in f.read().splitlines() if line.strip()]
            
    return G_sub, model_xgb, model_rf, feature_cols

def get_traffic_factor_ui(highway_type, hour):
    """Simple replica of traffic calculation for real-time predictions."""
    base_demand = {
        'motorway': 0.85, 'trunk': 0.80, 'primary': 0.75,
        'secondary': 0.50, 'tertiary': 0.30, 'residential': 0.10
    }
    if isinstance(highway_type, list):
        highway_type = highway_type[0]
    demand = base_demand.get(highway_type, 0.15)
    
    if 7 <= hour <= 9: diurnal = 3.2
    elif 16 <= hour <= 18: diurnal = 3.5
    elif 10 <= hour <= 15: diurnal = 1.3
    else: diurnal = 0.5
    
    return float(round(1.0 + 3.0 * np.clip(demand * diurnal, 0.0, 1.0), 2))

def parse_sinuosity_ui(data, length):
    if "geometry" in data:
        try:
            geom = data["geometry"]
            if isinstance(geom, str):
                if geom.startswith("LINESTRING"):
                    coords_str = geom.replace("LINESTRING (", "").replace(")", "")
                    coords = [list(map(float, pt.split())) for pt in coords_str.split(", ")]
                else:
                    coords = []
            else:
                coords = list(geom.coords)
                
            if len(coords) >= 2:
                p1 = coords[0]
                pn = coords[-1]
                dy = (pn[1] - p1[1]) * 111000.0
                dx = (pn[0] - p1[0]) * 68000.0
                straight = np.sqrt(dx**2 + dy**2)
                if straight > 0:
                    return float(np.clip(length / straight, 1.0, 3.5))
        except:
            pass
    return 1.0

def predict_edge_difficulties(G, driver, hour, model, feature_cols, sensitivities=None):
    """Predict personalized difficulty for every edge in the graph dynamically."""
    edges_list = []
    keys_list = []
    
    # Vehicle specifications based on class
    v_class = driver["class"]
    v_width = 1.7 if v_class == 'Compact' else 1.9 if v_class == 'Sedan' else 2.1 if v_class == 'SUV' else 2.5
    v_weight = 1200 if v_class == 'Compact' else 1600 if v_class == 'Sedan' else 2200 if v_class == 'SUV' else 4500
    
    # 1. Compile raw feature rows for all edges
    for u, v, k, data in G.edges(keys=True, data=True):
        road_class = data.get("highway", "residential")
        if isinstance(road_class, list):
            road_class = road_class[0]
            
        length = float(data.get("length", 10.0))
        slope = float(data.get("slope", 0.0))
        is_one_way = 1 if data.get("oneway", "False") == "True" else 0
        surface = data.get("surface", "asphalt")
        if isinstance(surface, list):
            surface = surface[0]
            
        is_lit = 1 if data.get("lit", "yes") == "yes" else 0
        has_tram = 1 if ("railway" in data or data.get("railway") == "tram") else 0
        
        lanes_val = data.get("lanes", "1")
        if isinstance(lanes_val, list):
            lanes_val = lanes_val[0]
        try:
            lanes = int(float(str(lanes_val)))
        except ValueError:
            lanes = 1
            
        width_val = data.get("width", None)
        if isinstance(width_val, list):
            width_val = width_val[0]
        try:
            road_width = float(width_val) if width_val else None
        except:
            road_width = None
            
        if not road_width:
            width_defaults = {
                "motorway": 12.0, "primary": 8.5, "secondary": 7.0,
                "tertiary": 6.0, "residential": 4.5, "living_street": 3.5
            }
            road_width = width_defaults.get(road_class, 5.0)
            
        sinuosity = parse_sinuosity_ui(data, length)
        traffic_factor = get_traffic_factor_ui(road_class, hour)
        
        # caution speed estimation
        speed_limit = 50.0
        maxspeed_val = data.get("maxspeed", "50")
        if isinstance(maxspeed_val, list):
            maxspeed_val = maxspeed_val[0]
        try:
            speed_limit = float(maxspeed_val.replace("DE:urban", "50").replace("DE:zone:30", "30"))
        except:
            pass
        actual_speed = float(round(np.clip(speed_limit * (1.0 / traffic_factor), 5.0, speed_limit), 1))
        
        edges_list.append({
            # Driver profile
            "driver_age": driver["age"],
            "driver_experience": driver["experience"],
            "vehicle_width": v_width,
            "vehicle_weight": v_weight,
            "highway_comfort": driver["highway_comfort"],
            "narrow_road_comfort": driver["narrow_road_comfort"],
            "night_vision": driver["night_vision"],
            # Segment features
            "hour": hour,
            "is_night": 1 if (hour >= 20 or hour <= 6) else 0,
            "length": length,
            "slope": slope,
            "is_lit": is_lit,
            "has_tram": has_tram,
            "is_one_way": is_one_way,
            "lanes": lanes,
            "sinuosity": sinuosity,
            "traffic_factor": traffic_factor,
            "actual_speed": actual_speed,
            "road_width": road_width,
            "road_class": road_class,
            "vehicle_class": v_class,
            "surface": surface
        })
        keys_list.append((u, v, k))
        
    df_edges = pd.DataFrame(edges_list)
    
    # 2. One-hot encoding matching train feature set
    df_encoded = pd.get_dummies(df_edges, columns=['road_class', 'vehicle_class', 'surface'], drop_first=False)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
            
    # Calculate interaction features
    df_encoded['width_margin'] = df_encoded['road_width'] - df_encoded['vehicle_width']
    df_encoded['experience_traffic'] = df_encoded['driver_experience'] * df_encoded['traffic_factor']
    df_encoded['narrow_comfort_margin'] = df_encoded['narrow_road_comfort'] * df_encoded['width_margin']
    df_encoded['night_vision_lit'] = df_encoded['night_vision'] * df_encoded['is_lit'] * df_encoded['is_night']
    df_encoded['experience_sinuosity'] = df_encoded['driver_experience'] * df_encoded['sinuosity']
    df_encoded['weight_slope'] = (df_encoded['vehicle_weight'] / 1000.0) * df_encoded['slope']
    
    # Align features with the exact column list from training
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    X_pred = df_encoded[feature_cols]
    
    # 3. Model inference
    if model:
        predictions = model.predict(X_pred)
    else:
        # Fallback to simple gradient-only baseline
        predictions = 1.0 + np.abs(df_edges['slope']) * 0.4
        
    # Apply user-defined hazard sensitivities
    if sensitivities:
        # We need a mutable array for predictions
        predictions = np.array(predictions, dtype=float)
        for idx in range(len(predictions)):
            row = df_edges.iloc[idx]
            score = float(predictions[idx])
            
            # Extract attributes
            slope = abs(float(row["slope"]))
            road_width = float(row["road_width"])
            vehicle_width = float(row["vehicle_width"])
            width_margin = road_width - vehicle_width
            sinuosity = float(row["sinuosity"])
            traffic_factor = float(row["traffic_factor"])
            is_night = int(row["is_night"])
            is_lit = int(row["is_lit"])
            surface = str(row["surface"])
            lanes = int(row["lanes"])
            road_class = str(row["road_class"])
            actual_speed = float(row["actual_speed"])
            
            # Retrieve graph edge data directly for additional OSM tags
            u, v, k = keys_list[idx]
            data = G[u].get(v, {}).get(k, {})
            
            # Adjust score based on sensitivities
            
            # 1. Slope (Average Gradient)
            if slope > 3.0:
                score += (slope / 3.0) * 0.3 * (sensitivities["slope"] - 1.0)
            
            # 2. Max Gradient (approx as slope * 1.2)
            max_slope = slope * 1.2
            if max_slope > 4.0:
                score += (max_slope / 4.0) * 0.2 * (sensitivities["max_slope"] - 1.0)
                
            # 3. Sinuosity
            if sinuosity > 1.1:
                score += (sinuosity - 1.0) * 1.5 * (sensitivities["sinuosity"] - 1.0)
                
            # 4. Curve Density (curves per km)
            curve_dens = (sinuosity - 1.0) * 10.0
            if curve_dens > 1.0:
                score += curve_dens * 0.2 * (sensitivities["curve_density"] - 1.0)
                
            # 5. Lanes (stress when fewer lanes exist)
            if lanes <= 1:
                score += 0.5 * (sensitivities["lanes"] - 1.0)
                
            # 6. Road Width (narrowness stress)
            if width_margin < 1.0:
                score += (1.0 / max(0.1, width_margin)) * (sensitivities["narrow"] - 1.0)
                
            # 7. Has Median Divider (dual_carriageway or tag check)
            has_median = 1 if (road_class in ["motorway", "trunk"] or data.get("dual_carriageway") == "yes") else 0
            if not has_median:
                score += 0.3 * (sensitivities["median"] - 1.0)
                
            # 8. Is One-Way
            is_one_way = (row["is_one_way"] == 1)
            if not is_one_way and road_width < 5.5:
                score += 0.8 * (sensitivities["oneway"] - 1.0)
                
            # 9. Road Class overrides
            if road_class == "motorway":
                score += 0.5 * (sensitivities["class_motorway"] - 1.0)
            elif road_class == "primary":
                score += 0.3 * (sensitivities["class_primary"] - 1.0)
            elif road_class == "secondary":
                score += 0.2 * (sensitivities["class_secondary"] - 1.0)
            elif road_class == "tertiary":
                score += 0.1 * (sensitivities["class_tertiary"] - 1.0)
            elif road_class == "residential":
                score += 0.2 * (sensitivities["class_residential"] - 1.0)
                
            # 10. Surface Type
            if surface == "cobblestone":
                score += 0.8 * (sensitivities["surface_cobble"] - 1.0)
                
            # 11. Smoothness Index (rough pavement)
            smoothness = data.get("smoothness", "good")
            if smoothness in ["bad", "very_bad", "rough", "cobblestone"]:
                score += 0.6 * (sensitivities["smoothness"] - 1.0)
                
            # 12. Is Lit (night driving unlit stress)
            if is_night and not is_lit:
                score += 1.5 * (sensitivities["night_unlit"] - 1.0)
                
            # 13. Has Tunnel
            has_tunnel = 1 if ("tunnel" in data and data["tunnel"] != "no") else 0
            if has_tunnel:
                score += 1.0 * (sensitivities["tunnel"] - 1.0)
                
            # 14. Has Bridge
            has_bridge = 1 if ("bridge" in data and data["bridge"] != "no") else 0
            if has_bridge:
                score += 0.6 * (sensitivities["bridge"] - 1.0)
                
            # 15. Junction Density
            junc_density = 2.0 if road_class in ["residential", "living_street"] else 0.5
            score += junc_density * 0.15 * (sensitivities["junction_density"] - 1.0)
            
            # 16. Has Roundabout
            has_roundabout = 1 if (data.get("junction") == "roundabout") else 0
            if has_roundabout:
                score += 0.8 * (sensitivities["roundabout"] - 1.0)
                
            # 17. Stop Sign Count (approximated)
            stop_signs = 1 if (road_class == "residential" and not is_one_way) else 0
            if stop_signs:
                score += 0.4 * (sensitivities["stops"] - 1.0)
                
            # 18. Traffic Light Count
            traffic_lights = 1 if (road_class in ["primary", "secondary"] and data.get("junction") == "intersection") else 0
            if traffic_lights:
                score += 0.5 * (sensitivities["signals"] - 1.0)
                
            # 19. Has Speed Bump
            has_bump = 1 if (data.get("traffic_calming") == "yes" or road_class == "living_street") else 0
            if has_bump:
                score += 0.5 * (sensitivities["speed_bumps"] - 1.0)
                
            # 20. Has Tram Tracks
            if has_tram:
                score += 0.8 * (sensitivities["tram_tracks"] - 1.0)
                
            # 21. Has Cycleway
            has_cycle = 1 if ("cycleway" in data or data.get("cycleway") == "yes") else 0
            if has_cycle:
                score += 0.6 * (sensitivities["cycleway"] - 1.0)
                
            # 22. Pedestrian Crossing Density
            has_crossing = 1 if (data.get("crossing") == "yes" or road_class in ["primary", "secondary"]) else 0
            score += has_crossing * 0.4 * (sensitivities["pedestrian_crossing"] - 1.0)
            
            # 23. Has On-Street Parking
            has_parking = 1 if (road_class == "residential" and not is_one_way) else 0
            if has_parking:
                score += 0.5 * (sensitivities["parking"] - 1.0)
                
            # 24. Is Urban (always 1 for Berlin Mitte)
            score += 0.4 * (sensitivities["urban"] - 1.0)
            
            # 25. Is Forest/Jungle (Tiergarten park roads)
            name_val = data.get("name", "")
            if isinstance(name_val, list):
                name_val = " ".join([str(n) for n in name_val])
            is_forest = 1 if ("tiergarten" in str(name_val).lower() or road_class == "living_street") else 0
            if is_forest:
                score += 0.5 * (sensitivities["forest"] - 1.0)
                
            # 26. Is Mountainous (Berlin is flat, but simulated based on slopes)
            is_mountain = 1 if abs(slope) > 4.0 else 0
            if is_mountain:
                score += 0.6 * (sensitivities["mountain"] - 1.0)
                
            # 27. Is Rural Open (approximated for edge boundaries)
            is_rural = 1 if (road_class == "unclassified") else 0
            if is_rural:
                score += 0.4 * (sensitivities["rural"] - 1.0)
                
            # 28. Speed Limit
            if actual_speed > 50.0:
                score += ((actual_speed - 50.0) / 50.0) * 0.5 * (sensitivities["speed_limit"] - 1.0)
                
            # 29. Traffic Congestion (traffic factor)
            if traffic_factor > 1.5:
                score += (traffic_factor - 1.0) * 0.4 * (sensitivities["traffic"] - 1.0)
                
            # 30. Lane Change Density
            lane_changes = 1 if (lanes >= 3 and road_class in ["motorway", "primary"]) else 0
            if lane_changes:
                score += 0.6 * (sensitivities["lane_change_density"] - 1.0)
                
            predictions[idx] = score
            
    predictions = np.clip(predictions, 1.0, 5.0)
    
    # Write predicted difficulty back to graph edge attributes
    edge_difficulties = {}
    for idx, (u, v, k) in enumerate(keys_list):
        score = float(predictions[idx])
        G[u][v][k]["predicted_difficulty"] = score
        
        # Define personalized routing cost
        # Combined weight: physical distance * predicted difficulty
        G[u][v][k]["personalized_cost"] = G[u][v][k]["length"] * (score ** 2)
        edge_difficulties[(u, v, k)] = score
        
    return edge_difficulties

def get_route_color(score):
    """Map difficulty score (1.0 - 5.0) to a Hex Color code (Green -> Yellow -> Red)."""
    if score < 2.0:
        return "#17b978"  # Green
    elif score < 3.2:
        return "#ffc045"  # Yellow/Orange
    else:
        return "#ff4b4b"  # Red

def compile_route_details(path, G_net, driver, hour):
    rows = []
    v_class = driver["class"]
    v_width = 1.7 if v_class == 'Compact' else 1.9 if v_class == 'Sedan' else 2.1 if v_class == 'SUV' else 2.5
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        data = G_net.get_edge_data(u, v)[0]
        
        street_name = data.get("name", "Unnamed Street")
        if isinstance(street_name, list):
            street_name = " / ".join([str(s) for s in street_name])
            
        length = float(data.get("length", 10.0))
        road_class = data.get("highway", "residential")
        if isinstance(road_class, list):
            road_class = road_class[0]
            
        slope = float(data.get("slope", 0.0))
        surface = data.get("surface", "asphalt")
        if isinstance(surface, list):
            surface = surface[0]
            
        width_val = data.get("width", None)
        if isinstance(width_val, list):
            width_val = width_val[0]
        try:
            road_width = float(width_val) if width_val else None
        except:
            road_width = None
        if not road_width:
            width_defaults = {
                "motorway": 12.0, "primary": 8.5, "secondary": 7.0,
                "tertiary": 6.0, "residential": 4.5, "living_street": 3.5
            }
            road_width = width_defaults.get(road_class, 5.0)
            
        difficulty = float(data.get("predicted_difficulty", 1.0))
        traffic_factor = get_traffic_factor_ui(road_class, hour)
        
        # Determine stress reasons
        reasons = []
        if road_width - v_width < 1.0:
            reasons.append("Narrow road margin")
        if abs(slope) > 4.0:
            reasons.append(f"Steep incline ({slope:.1f}%)")
        if surface == "cobblestone":
            reasons.append("Cobblestone surface")
        if traffic_factor > 2.0:
            reasons.append(f"Congestion ({traffic_factor:.1f}x traffic)")
        if "railway" in data or data.get("railway") == "tram":
            reasons.append("Tram tracks on road")
            
        reasons_str = ", ".join(reasons) if reasons else "Clear & smooth flow"
        
        rows.append({
            "Segment": i + 1,
            "Street Name": street_name,
            "Class": road_class.capitalize(),
            "Length (m)": int(length),
            "Slope (%)": f"{slope:.1f}%",
            "Surface": surface.capitalize(),
            "Width (m)": f"{road_width:.1f}m",
            "Difficulty": float(round(difficulty, 2)),
            "Key Stress Factors": reasons_str
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def geocode_address(query):
    """Geocode address query using Nominatim with a User-Agent."""
    if not query:
        return None
    # Auto-append Berlin if not present to search locally
    search_query = query
    if "berlin" not in query.lower():
        search_query = f"{query}, Berlin, Germany"
        
    try:
        headers = {"User-Agent": "BerlinRouteDifficultyThesisApp/1.0 (contact: seena.student@hu-berlin.de)"}
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": search_query, "format": "json", "limit": 1}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                display_name = data[0]["display_name"].split(",")[0]
                return lat, lon, display_name
    except Exception as e:
        pass
    return None

def main():
    st.title("🚗 Personalized Navigation & Route Difficulty Predictor")
    st.caption("Master's Thesis Demonstration Area - Berlin Mitte, Germany")
    
    # Load resources
    G, model_xgb, model_rf, feature_cols = load_resources()
    
    # --- SIDEBAR: DRIVER CONFIGURATION PANEL ---
    st.sidebar.header("👤 Driver & Vehicle Profile")
    
    driver_age = st.sidebar.slider("Driver Age", 18, 85, 30)
    
    max_exp = max(0, driver_age - 18)
    driver_exp = st.sidebar.slider("Driving Experience (Years)", 0, min(max_exp, 50), 10)
    
    vehicle_class = st.sidebar.selectbox(
        "Vehicle Category",
        ["Compact", "Sedan", "SUV", "RV / Caravan"]
    )
    # Simplify class tag to match model categories
    v_class = "RV" if "RV" in vehicle_class else vehicle_class
    v_width = 1.7 if v_class == 'Compact' else 1.9 if v_class == 'Sedan' else 2.1 if v_class == 'SUV' else 2.5
    v_weight = 1200 if v_class == 'Compact' else 1600 if v_class == 'Sedan' else 2200 if v_class == 'SUV' else 4500
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Environment Settings")
    
    dep_hour = st.sidebar.slider("Departure Hour (Traffic peak hours: 8 & 17)", 0, 23, 14)
    
    # Check which models loaded successfully to handle missing files on cloud deploys
    available_models = []
    if model_xgb is not None:
        available_models.append("XGBoost Regressor (Personalized)")
    if model_rf is not None:
        available_models.append("Random Forest (Personalized)")
    available_models.append("Heuristic Baseline (Non-Personalized)")
    
    if not available_models:
        available_models = ["Heuristic Baseline (Non-Personalized)"]
        
    model_type = st.sidebar.radio(
        "Predictive Model Selection",
        available_models
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Custom Sensitivities for All 30 Thesis Features")
    st.sidebar.caption("Override/adjust the stress weights (1.0 = AI Default):")

    with st.sidebar.expander("📐 A. Road Geometry & Lanes"):
        sens_slope = st.slider("Average Gradient (Slopes)", 0.0, 3.0, 1.0, 0.1)
        sens_max_slope = st.slider("Max Gradient", 0.0, 3.0, 1.0, 0.1)
        sens_sinuosity = st.slider("Sinuosity (Windingness)", 0.0, 3.0, 1.0, 0.1)
        sens_curve_density = st.slider("Curve Density", 0.0, 3.0, 1.0, 0.1)
        sens_lanes = st.slider("Lane Count (Fewer Lanes)", 0.0, 3.0, 1.0, 0.1)
        sens_narrow = st.slider("Road Width (Narrowness)", 0.0, 3.0, 1.0, 0.1)
        sens_median = st.slider("Lack of Median Divider", 0.0, 3.0, 1.0, 0.1)
        sens_oneway = st.slider("One-Way vs Two-Way", 0.0, 3.0, 1.0, 0.1)

    with st.sidebar.expander("🧱 B. Infrastructure & Surfaces"):
        sens_surface_cobble = st.slider("Cobblestone Surfaces", 0.0, 3.0, 1.0, 0.1)
        sens_smoothness = st.slider("Pavement Roughness", 0.0, 3.0, 1.0, 0.1)
        sens_night_unlit = st.slider("Is Lit (Unlit Night Streets)", 0.0, 3.0, 1.0, 0.1)
        sens_tunnel = st.slider("Tunnel Stress", 0.0, 3.0, 1.0, 0.1)
        sens_bridge = st.slider("Bridge Stress", 0.0, 3.0, 1.0, 0.1)

    with st.sidebar.expander("🚦 C. Junctions, Traffic & Speed"):
        sens_junction_density = st.slider("Junction Density", 0.0, 3.0, 1.0, 0.1)
        sens_roundabout = st.slider("Roundabouts", 0.0, 3.0, 1.0, 0.1)
        sens_stops = st.slider("Stop Sign Count", 0.0, 3.0, 1.0, 0.1)
        sens_signals = st.slider("Traffic Light Count", 0.0, 3.0, 1.0, 0.1)
        sens_speed_bumps = st.slider("Speed Bumps", 0.0, 3.0, 1.0, 0.1)
        sens_speed_limit = st.slider("Speed Limit (High Speed)", 0.0, 3.0, 1.0, 0.1)
        sens_traffic = st.slider("Traffic Congestion", 0.0, 3.0, 1.0, 0.1)
        sens_lane_change_density = st.slider("Lane Change Density", 0.0, 3.0, 1.0, 0.1)

    with st.sidebar.expander("⚠️ D. Sharing, Environment & Class"):
        sens_tram_tracks = st.slider("Tram Tracks", 0.0, 3.0, 1.0, 0.1)
        sens_cycleway = st.slider("Adjacent Cycleways", 0.0, 3.0, 1.0, 0.1)
        sens_pedestrian_crossing = st.slider("Pedestrian Crossings", 0.0, 3.0, 1.0, 0.1)
        sens_parking = st.slider("On-Street Parking", 0.0, 3.0, 1.0, 0.1)
        sens_urban = st.slider("Is Urban driving", 0.0, 3.0, 1.0, 0.1)
        sens_forest = st.slider("Is Forest / Canopy", 0.0, 3.0, 1.0, 0.1)
        sens_mountain = st.slider("Is Mountainous", 0.0, 3.0, 1.0, 0.1)
        sens_rural = st.slider("Is Rural Open", 0.0, 3.0, 1.0, 0.1)
        sens_class_motorway = st.slider("Motorway Class", 0.0, 3.0, 1.0, 0.1)
        sens_class_primary = st.slider("Primary Road Class", 0.0, 3.0, 1.0, 0.1)
        sens_class_secondary = st.slider("Secondary Road Class", 0.0, 3.0, 1.0, 0.1)
        sens_class_tertiary = st.slider("Tertiary Road Class", 0.0, 3.0, 1.0, 0.1)
        sens_class_residential = st.slider("Residential Road Class", 0.0, 3.0, 1.0, 0.1)
        
    sensitivities = {
        "slope": sens_slope,
        "max_slope": sens_max_slope,
        "sinuosity": sens_sinuosity,
        "curve_density": sens_curve_density,
        "lanes": sens_lanes,
        "narrow": sens_narrow,
        "median": sens_median,
        "oneway": sens_oneway,
        "surface_cobble": sens_surface_cobble,
        "smoothness": sens_smoothness,
        "night_unlit": sens_night_unlit,
        "tunnel": sens_tunnel,
        "bridge": sens_bridge,
        "junction_density": sens_junction_density,
        "roundabout": sens_roundabout,
        "stops": sens_stops,
        "signals": sens_signals,
        "speed_bumps": sens_speed_bumps,
        "speed_limit": sens_speed_limit,
        "traffic": sens_traffic,
        "tram_tracks": sens_tram_tracks,
        "cycleway": sens_cycleway,
        "pedestrian_crossing": sens_pedestrian_crossing,
        "parking": sens_parking,
        "urban": sens_urban,
        "forest": sens_forest,
        "mountain": sens_mountain,
        "rural": sens_rural,
        "class_motorway": sens_class_motorway,
        "class_primary": sens_class_primary,
        "class_secondary": sens_class_secondary,
        "class_tertiary": sens_class_tertiary,
        "class_residential": sens_class_residential,
        "lane_change_density": sens_lane_change_density
    }
    
    # Assemble driver dict
    driver = {
        "age": driver_age,
        "experience": driver_exp,
        "class": v_class,
        # Comfort estimators based on experience
        "highway_comfort": 0.9 if driver_exp > 15 else 0.7 if driver_exp > 2 else 0.4,
        "narrow_road_comfort": 0.8 if driver_exp > 10 else 0.6 if driver_exp > 2 else 0.3,
        "night_vision": 1.0 - (0.01 * max(0, driver_age - 50))
    }
    
    # Choose active model
    active_model = None
    if "XGBoost" in model_type:
        active_model = model_xgb
    elif "Random Forest" in model_type:
        active_model = model_rf
        
    # --- CALC DYNAMIC DIFFICULTY ON ROAD NETWORK ---
    with st.spinner("Calculating personalized road stress across Berlin Mitte..."):
        predict_edge_difficulties(G, driver, dep_hour, active_model, feature_cols, sensitivities)
        
    # --- MAIN PANEL: ROUTING SELECTION ---
    st.markdown("### 🗺️ Route Start & End Point Selection")
    input_mode = st.radio(
        "Select Location Input Mode",
        ["📍 Preset Landmarks", "🔍 Custom Address Search", "🌐 Coordinate Input", "🎲 Random Graph Nodes"],
        horizontal=True
    )
    
    # Track random nodes in session state
    if "rand_start_node" not in st.session_state:
        st.session_state.rand_start_node = None
    if "rand_end_node" not in st.session_state:
        st.session_state.rand_end_node = None
        
    start_coords = None
    end_coords = None
    start_place = ""
    end_place = ""
    
    if input_mode == "📍 Preset Landmarks":
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            start_place = st.selectbox("Start Point", list(LANDMARKS.keys()), index=2) # Hauptbahnhof
        with col_sel2:
            end_place = st.selectbox("End Point", list(LANDMARKS.keys()), index=1) # Alexanderplatz
        start_coords = LANDMARKS[start_place]
        end_coords = LANDMARKS[end_place]
        
    elif input_mode == "🔍 Custom Address Search":
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            start_query = st.text_input("Search Start Location", "Hauptbahnhof, Berlin")
        with col_sel2:
            end_query = st.text_input("Search End Location", "Alexanderplatz, Berlin")
            
        start_res = geocode_address(start_query)
        end_res = geocode_address(end_query)
        
        if start_res is None:
            st.error(f"Could not resolve start address: '{start_query}'. Using fallback.")
            start_coords = LANDMARKS["Hauptbahnhof (Central Station)"]
            start_place = "Hauptbahnhof (Fallback)"
        else:
            start_coords = (start_res[0], start_res[1])
            start_place = start_res[2]
            
        if end_res is None:
            st.error(f"Could not resolve end address: '{end_query}'. Using fallback.")
            end_coords = LANDMARKS["Alexanderplatz (TV Tower)"]
            end_place = "Alexanderplatz (Fallback)"
        else:
            end_coords = (end_res[0], end_res[1])
            end_place = end_res[2]
            
    elif input_mode == "🌐 Coordinate Input":
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            start_lat = st.number_input("Start Latitude", value=52.525010, format="%.6f")
            start_lon = st.number_input("Start Longitude", value=13.369402, format="%.6f")
        with col_sel2:
            end_lat = st.number_input("End Latitude", value=52.520815, format="%.6f")
            end_lon = st.number_input("End Longitude", value=13.409419, format="%.6f")
        start_coords = (start_lat, start_lon)
        end_coords = (end_lat, end_lon)
        start_place = f"Custom ({start_lat:.4f}, {start_lon:.4f})"
        end_place = f"Custom ({end_lat:.4f}, {end_lon:.4f})"
        
    elif input_mode == "🎲 Random Graph Nodes":
        all_nodes = list(G.nodes())
        if st.session_state.rand_start_node is None or st.session_state.rand_start_node not in G:
            st.session_state.rand_start_node = random.choice(all_nodes)
        if st.session_state.rand_end_node is None or st.session_state.rand_end_node not in G:
            st.session_state.rand_end_node = random.choice(all_nodes)
            while st.session_state.rand_end_node == st.session_state.rand_start_node:
                st.session_state.rand_end_node = random.choice(all_nodes)
                
        col_rand1, col_rand2 = st.columns([3, 1])
        with col_rand1:
            st.markdown(f"📍 **Start Point**: Node `{st.session_state.rand_start_node}`")
            st.markdown(f"🏁 **End Point**: Node `{st.session_state.rand_end_node}`")
        with col_rand2:
            if st.button("🎲 Generate New Random Locations", use_container_width=True):
                st.session_state.rand_start_node = random.choice(all_nodes)
                st.session_state.rand_end_node = random.choice(all_nodes)
                while st.session_state.rand_end_node == st.session_state.rand_start_node:
                    st.session_state.rand_end_node = random.choice(all_nodes)
                st.rerun()
                
        start_coords = (G.nodes[st.session_state.rand_start_node]['y'], G.nodes[st.session_state.rand_start_node]['x'])
        end_coords = (G.nodes[st.session_state.rand_end_node]['y'], G.nodes[st.session_state.rand_end_node]['x'])
        start_place = f"Random Node {st.session_state.rand_start_node}"
        end_place = f"Random Node {st.session_state.rand_end_node}"
        
    # Check boundaries warning
    lats = [data['y'] for node, data in G.nodes(data=True)]
    lons = [data['x'] for node, data in G.nodes(data=True)]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    margin = 0.03
    for lat, lon, name in [(start_coords[0], start_coords[1], "Start"), (end_coords[0], end_coords[1], "End")]:
        if not ((min_lat - margin <= lat <= max_lat + margin) and (min_lon - margin <= lon <= max_lon + margin)):
            st.warning(f"⚠️ **Warning**: The {name} point ({lat:.4f}, {lon:.4f}) is outside our active Berlin graph bounding box. We will snap it to the nearest valid intersection.")
            
    if start_coords == end_coords:
        st.error("Start and End locations must be different.")
        return
        
    # Find nearest nodes
    start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
    
    # Compute routing paths
    try:
        # Route 1: Shortest Distance
        path_distance = nx.shortest_path(G, start_node, end_node, weight="length")
        
        # Route 2: Comfort-Optimized (predicted difficulty cost)
        path_comfort = nx.shortest_path(G, start_node, end_node, weight="personalized_cost")
        
        # Route 3: Alternative Bypass (penalizing edges of Route 1 & 2)
        G_alt = G.copy()
        edges_to_penalize = set()
        for path in [path_distance, path_comfort]:
            for idx in range(len(path) - 1):
                edges_to_penalize.add((path[idx], path[idx+1]))
                edges_to_penalize.add((path[idx+1], path[idx]))
                
        for u, v, k in G_alt.edges(keys=True):
            if (u, v) in edges_to_penalize or (v, u) in edges_to_penalize:
                G_alt[u][v][k]["personalized_cost"] *= 4.0
                G_alt[u][v][k]["length"] *= 4.0
                
        try:
            path_bypass = nx.shortest_path(G_alt, start_node, end_node, weight="personalized_cost")
        except:
            path_bypass = path_comfort
            
    except nx.NetworkXNoPath:
        st.error("Could not find a valid path between the selected coordinates.")
        return
        
    # Helper to calculate route statistics
    def get_path_metrics(path, G_net, hour):
        length = 0
        diffs = []
        travel_time_sec = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # Get edge data dictionary
            data = G_net.get_edge_data(u, v)[0]
            length += float(data.get("length", 10.0))
            score = float(data.get("predicted_difficulty", 1.0))
            diffs.append(score)
            
            road_class = data.get("highway", "residential")
            if isinstance(road_class, list):
                road_class = road_class[0]
            traffic = get_traffic_factor_ui(road_class, hour)
            
            # Simple travel time: length / (speed_limit_mps / traffic)
            maxspeed_val = data.get("maxspeed", "50")
            if isinstance(maxspeed_val, list):
                maxspeed_val = maxspeed_val[0]
            try:
                speed_limit = float(maxspeed_val.replace("DE:urban", "50").replace("DE:zone:30", "30"))
            except:
                speed_limit = 50.0
            actual_speed_mps = (speed_limit / 3.6) / traffic
            travel_time_sec += float(data.get("length", 10.0)) / max(1.0, actual_speed_mps)
            
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        if avg_diff < 2.0:
            label = "🟢 Easy Comfort"
        elif avg_diff < 3.2:
            label = "🟡 Moderate Stress"
        else:
            label = "🔴 Hard/Difficult"
            
        return {
            "length_km": length / 1000.0,
            "avg_diff": avg_diff,
            "max_diff": max_diff,
            "time_min": travel_time_sec / 60.0,
            "label": label,
            "path": path
        }
        
    metrics_distance = get_path_metrics(path_distance, G, dep_hour)
    metrics_comfort = get_path_metrics(path_comfort, G, dep_hour)
    metrics_bypass = get_path_metrics(path_bypass, G, dep_hour)
    
    # --- ROUTE COMPARISON DASHBOARD ---
    st.markdown("### 📊 Route Options Comparison Table")
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    
    with col_comp1:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #7f8c8d; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <h4>Option 1: Shortest Path</h4>
            <p><b>Difficulty Label</b>: {metrics_distance['label']}</p>
            <p>🛣️ <b>Distance</b>: {metrics_distance['length_km']:.2f} km</p>
            <p>⏱️ <b>Est. Time</b>: {metrics_distance['time_min']:.1f} mins</p>
            <p>⚠️ <b>Avg Stress</b>: {metrics_distance['avg_diff']:.2f} / 5.0</p>
            <p>🔥 <b>Max Bottleneck</b>: {metrics_distance['max_diff']:.2f} / 5.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_comp2:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #17b978; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <h4>Option 2: Comfort-Optimized</h4>
            <p><b>Difficulty Label</b>: {metrics_comfort['label']}</p>
            <p>🛣️ <b>Distance</b>: {metrics_comfort['length_km']:.2f} km</p>
            <p>⏱️ <b>Est. Time</b>: {metrics_comfort['time_min']:.1f} mins</p>
            <p>⚠️ <b>Avg Stress</b>: {metrics_comfort['avg_diff']:.2f} / 5.0</p>
            <p>🔥 <b>Max Bottleneck</b>: {metrics_comfort['max_diff']:.2f} / 5.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_comp3:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc045; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <h4>Option 3: Alternative Bypass</h4>
            <p><b>Difficulty Label</b>: {metrics_bypass['label']}</p>
            <p>🛣️ <b>Distance</b>: {metrics_bypass['length_km']:.2f} km</p>
            <p>⏱️ <b>Est. Time</b>: {metrics_bypass['time_min']:.1f} mins</p>
            <p>⚠️ <b>Avg Stress</b>: {metrics_bypass['avg_diff']:.2f} / 5.0</p>
            <p>🔥 <b>Max Bottleneck</b>: {metrics_bypass['max_diff']:.2f} / 5.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("")
    selected_opt = st.radio(
        "🎯 Select Route Option to Highlight and Analyze Detailed Streets:",
        ["Option 1: Shortest Path", "Option 2: Comfort-Optimized Path", "Option 3: Alternative Bypass Path"],
        horizontal=True,
        index=1 # Default to Comfort path
    )
    
    # Set active path
    if "Option 1" in selected_opt:
        active_path = path_distance
        active_label = "Shortest Path"
    elif "Option 2" in selected_opt:
        active_path = path_comfort
        active_label = "Comfort-Optimized"
    else:
        active_path = path_bypass
        active_label = "Alternative Bypass"
        
    # --- RENDER MAPS WITH COLOR-CODED SEGMENTS ---
    # Center map on active route
    mid_node = active_path[len(active_path)//2]
    mid_lat = G.nodes[mid_node]['y']
    mid_lon = G.nodes[mid_node]['x']
    
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=14, tiles="cartodbpositron")
    
    # 1. Render start and end markers
    folium.Marker(
        location=start_coords,
        popup=f"Start: {start_place}",
        tooltip=f"Start: {start_place}",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    
    folium.Marker(
        location=end_coords,
        popup=f"End: {end_place}",
        tooltip=f"End: {end_place}",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)
    
    # Render all preset landmarks as blue reference points (if not close to start/end)
    for name, coords in LANDMARKS.items():
        dist_start = np.sqrt((coords[0] - start_coords[0])**2 + (coords[1] - start_coords[1])**2)
        dist_end = np.sqrt((coords[0] - end_coords[0])**2 + (coords[1] - end_coords[1])**2)
        if dist_start > 0.001 and dist_end > 0.001:
            folium.Marker(
                location=coords,
                popup=name,
                tooltip=name,
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
        
    # 2. Draw inactive routes as thin dashed lines for comparison
    inactive_paths = []
    if active_path is not path_distance:
        inactive_paths.append((path_distance, "Shortest Path Option", "#7f8c8d"))
    if active_path is not path_comfort:
        inactive_paths.append((path_comfort, "Comfort-Optimized Option", "#3498db"))
    if active_path is not path_bypass and path_bypass is not path_comfort:
        inactive_paths.append((path_bypass, "Alternative Bypass Option", "#9b59b6"))
        
    for path, label, color_code in inactive_paths:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G.get_edge_data(u, v)[0]
            
            if "geometry" in edge_data:
                geom = edge_data["geometry"]
                if isinstance(geom, str):
                    coords_str = geom.replace("LINESTRING (", "").replace(")", "")
                    points = [list(map(float, pt.split()))[::-1] for pt in coords_str.split(", ")]
                else:
                    points = [[lat, lon] for lon, lat in geom.coords]
            else:
                points = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
                
            folium.PolyLine(
                locations=points,
                color=color_code,
                weight=3,
                opacity=0.45,
                dash_array="5, 5",
                tooltip=label
            ).add_to(m)
            
    # 3. Draw active route, color-coded by predicted segment difficulty
    for i in range(len(active_path) - 1):
        u, v = active_path[i], active_path[i+1]
        edge_data = G.get_edge_data(u, v)[0]
        
        # Get line geometry coordinates
        if "geometry" in edge_data:
            geom = edge_data["geometry"]
            if isinstance(geom, str):
                coords_str = geom.replace("LINESTRING (", "").replace(")", "")
                points = [list(map(float, pt.split()))[::-1] for pt in coords_str.split(", ")]
            else:
                # Shapely LineString object
                points = [[lat, lon] for lon, lat in geom.coords]
        else:
            points = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
            
        score = edge_data.get("predicted_difficulty", 1.0)
        color = get_route_color(score)
        
        name_val = edge_data.get('name', 'Unnamed')
        if isinstance(name_val, list):
            name_val = " / ".join([str(n) for n in name_val])
            
        folium.PolyLine(
            locations=points,
            color=color,
            weight=7,
            opacity=0.9,
            tooltip=f"Active: {active_label} | Street: {name_val} | Difficulty: {score:.2f} | Slope: {float(edge_data.get('slope', 0.0)):.1f}%"
        ).add_to(m)
            
    # Display Map
    st.markdown(f"### 🗺️ Route Visual Map - Highlighting **{active_label}** (Green = Easy, Orange = Moderate, Red = Stressful)")
    folium_static(m, width=1100, height=550)
    
    # --- DRIVER CHOICE CONFIRMATION PANEL ---
    st.markdown("---")
    st.markdown("### 🗺️ Driver Choice Confirmation")
    
    # Selected metrics
    if "Option 1" in selected_opt:
        selected_metrics = metrics_distance
    elif "Option 2" in selected_opt:
        selected_metrics = metrics_comfort
    else:
        selected_metrics = metrics_bypass
        
    confirm_col1, confirm_col2 = st.columns([2, 1])
    
    with confirm_col1:
        st.markdown(f"You are currently analyzing the **{active_label}** route.")
        st.markdown(f"• **Overall Difficulty Index**: {selected_metrics['avg_diff']:.2f} / 5.0 ({selected_metrics['label']})")
        st.markdown(f"• **Total Travel Distance**: {selected_metrics['length_km']:.2f} km")
        st.markdown(f"• **Estimated Travel Time**: {selected_metrics['time_min']:.1f} mins")
        
        choice_btn = st.button(f"🚗 Accept and Launch Turn-by-Turn Navigation ({active_label})", use_container_width=True)
        if choice_btn:
            st.toast("Starting Navigation...", icon="🚗")
            st.balloons()
            st.success(f"🎉 **Navigation Started!** Sent route details for **{active_label}** to your in-car display. Safe travels!")
            
    with confirm_col2:
        st.metric(
            label="Overall Route Stress Score",
            value=f"{selected_metrics['avg_diff']:.2f} / 5.0",
            delta="Optimal Comfort" if "Comfort" in active_label else ("Non-Personalized" if "Shortest" in active_label else "Alternative Bypass"),
            delta_color="normal" if "Comfort" in active_label else "off"
        )
        
    # --- DETAILED SEGMENT ANALYSIS ---
    st.markdown("### 📋 Street-by-Street Route Stress Breakdown")
    with st.expander("🔍 View Detailed Street Segments & Hazard Analysis", expanded=True):
        df_details = compile_route_details(active_path, G, driver, dep_hour)
        
        # Color difficulty score column styling
        def color_difficulty(val):
            try:
                score = float(val)
                if score < 2.0:
                    color = '#d4edda' # Light green
                elif score < 3.2:
                    color = '#fff3cd' # Light yellow
                else:
                    color = '#f8d7da' # Light red
                return f'background-color: {color}'
            except:
                return ''
                
        # Format the table beautifully
        try:
            styled_df = df_details.style.map(color_difficulty, subset=["Difficulty"])
        except AttributeError:
            styled_df = df_details.style.applymap(color_difficulty, subset=["Difficulty"])
            
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
    
    # --- ACADEMIC VALUE EXPLANATION PANEL ---
    st.markdown("---")
    st.markdown("### 🎓 Thesis Concept Verification")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.info("""
            **How personalization changes routing**:
            When choosing **Personalized Driver Comfort**, the routing algorithm shifts from Dijkstra (minimizing physical meters) to minimizing a **personalized cost weight**.
            
            If a driver profile is highly vulnerable to a feature (e.g., an RV in narrow streets, or an elderly driver on dark roads), those segments represent a high penalty score, forcing the route to detour around them.
        """)
    with exp_col2:
        st.success(f"""
            **Current Active Profile Summary**:
            *   **Driver Class**: {driver_age}yo driver with {driver_exp} years of experience.
            *   **Vehicle Footprint**: Class '{v_class}' (Width: {v_width}m, Weight: {v_weight}kg).
            *   **Computed Baseline Compositions**:
                *   Highway Comfort: {driver['highway_comfort']:.2f}
                *   Narrow Road Comfort: {driver['narrow_road_comfort']:.2f}
                *   Night Vision Factor: {driver['night_vision']:.2f}
        """)

    # --- MODEL PERFORMANCE & EVALUATION SECTION ---
    st.markdown("---")
    st.markdown("### 🤖 Machine Learning Model Performance & Evaluation")
    
    tab_metrics, tab_importance = st.tabs(["📊 Model Evaluation Metrics", "📈 Feature Importances"])
    
    with tab_metrics:
        st.markdown("""
            **Academic evaluation results** of the personalized difficulty models across three split strategies:
            *   **Random Split**: Standard 80/20 train/test split.
            *   **User Split**: Testing on unseen drivers (proves personalization generalizes to new drivers).
            *   **Spatial Split**: Testing on unseen roads/neighborhoods (proves generalization to new geographical areas).
        """)
        
        metrics_csv_path = "data/processed/model_evaluation_metrics.csv"
        if os.path.exists(metrics_csv_path):
            df_metrics = pd.read_csv(metrics_csv_path)
            # Round numeric columns for cleaner display
            df_metrics["MAE"] = df_metrics["MAE"].round(3)
            df_metrics["RMSE"] = df_metrics["RMSE"].round(3)
            df_metrics["R2"] = df_metrics["R2"].round(3)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        else:
            st.warning("Model evaluation metrics file not found. Run model training first.")
            
    with tab_importance:
        st.markdown("**Relative weight of features** in determining personalized driving difficulty:")
        importance_img_path = "data/processed/feature_importances.png"
        if os.path.exists(importance_img_path):
            st.image(importance_img_path, caption="Feature Importances (XGBoost Regressor)", use_container_width=True)
        else:
            st.warning("Feature importance plot not found. Run model training first.")


if __name__ == "__main__":
    main()
