import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Config
LOGS_INPUT = "data/simulated_driving_logs.csv"
OUTPUT_DIR = "data/processed"

def engineer_features():
    """Load simulated driving logs, perform one-hot encoding, engineer interaction terms, and save splits."""
    print("--- Phase 4: Feature Engineering & Dataset Preparation ---")
    
    if not os.path.exists(LOGS_INPUT):
        raise FileNotFoundError(f"Missing driving logs at {LOGS_INPUT}. Please run driver_simulator.py first.")
        
    df = pd.read_csv(LOGS_INPUT)
    print(f"Loaded {len(df)} simulated edge-level logs.")
    
    # 1. Handle Categorical Road Classes
    # Map low-frequency classes to standard values
    class_map = {
        'motorway_link': 'motorway',
        'trunk_link': 'trunk',
        'primary_link': 'primary',
        'secondary_link': 'secondary',
        'tertiary_link': 'tertiary',
        'living_street': 'residential'
    }
    df['road_class'] = df['road_class'].replace(class_map)
    
    # One-hot encoding for road class, vehicle class, and surface
    df = pd.get_dummies(df, columns=['road_class', 'vehicle_class', 'surface'], drop_first=False)
    
    # Ensure boolean output from get_dummies is converted to 0/1 integers
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    # 2. Extract Road Width if not present and calculate Width Margin
    # Downloader uses typical widths if missing, which are passed in logs.
    # In case width is missing in logs, assign a default
    if 'road_width' not in df.columns:
        # We can approximate road_width from road class columns
        df['road_width'] = 6.0
        if 'road_class_motorway' in df.columns:
            df.loc[df['road_class_motorway'] == 1, 'road_width'] = 12.0
        if 'road_class_primary' in df.columns:
            df.loc[df['road_class_primary'] == 1, 'road_width'] = 8.5
        if 'road_class_secondary' in df.columns:
            df.loc[df['road_class_secondary'] == 1, 'road_width'] = 7.0
        if 'road_class_residential' in df.columns:
            df.loc[df['road_class_residential'] == 1, 'road_width'] = 4.5
            
    df['width_margin'] = df['road_width'] - df['vehicle_width']
    
    # 3. Create Interaction Features (Non-linear terms that help models learn personalization)
    df['experience_traffic'] = df['driver_experience'] * df['traffic_factor']
    df['narrow_comfort_margin'] = df['narrow_road_comfort'] * df['width_margin']
    df['night_vision_lit'] = df['night_vision'] * df['is_lit'] * df['is_night']
    df['experience_sinuosity'] = df['driver_experience'] * df['sinuosity']
    df['weight_slope'] = (df['vehicle_weight'] / 1000.0) * df['slope']
    
    print(f"Generated interaction features. Total features count: {len(df.columns)}")
    
    # 4. Save processed dataset for modeling
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "full_processed_dataset.csv"), index=False)
    
    # 5. Define Feature Sets for Training
    # We drop metadata/identifiers that cannot be fed to ML models
    drop_cols = ['driver_id', 'edge_u', 'edge_v', 'edge_key', 'difficulty_score']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print("\nTraining Features List:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
        
    # Write feature list to reference file
    with open(os.path.join(OUTPUT_DIR, "feature_list.txt"), "w") as f:
        f.write("\n".join(feature_cols))
        
    # --- Split Strategy A: Standard Random Split (80% Train, 20% Test) ---
    train_rand, test_rand = train_test_split(df, test_size=0.2, random_state=42)
    train_rand.to_csv(os.path.join(OUTPUT_DIR, "train_random.csv"), index=False)
    test_rand.to_csv(os.path.join(OUTPUT_DIR, "test_random.csv"), index=False)
    print(f"\nRandom Split: Train size={len(train_rand)}, Test size={len(test_rand)}")
    
    # --- Split Strategy B: User-Based Split (Hold out 20% of Drivers) ---
    # Evaluates how well the model predicts difficulty for a completely new driver
    all_drivers = list(df['driver_id'].unique())
    train_drivers, test_drivers = train_test_split(all_drivers, test_size=0.2, random_state=42)
    
    train_user = df[df['driver_id'].isin(train_drivers)]
    test_user = df[df['driver_id'].isin(test_drivers)]
    train_user.to_csv(os.path.join(OUTPUT_DIR, "train_user_split.csv"), index=False)
    test_user.to_csv(os.path.join(OUTPUT_DIR, "test_user_split.csv"), index=False)
    print(f"User-Based Split: Train drivers={len(train_drivers)}, Test drivers={len(test_drivers)}")
    print(f"                  Train size={len(train_user)}, Test size={len(test_user)}")
    
    # --- Split Strategy C: Spatial/Route-Based Split (Hold out 20% of Road Segments) ---
    # Evaluates how well the model predicts difficulty for an unseen road
    # Create unique edge keys
    df['edge_id'] = df['edge_u'].astype(str) + "_" + df['edge_v'].astype(str)
    all_edges = list(df['edge_id'].unique())
    train_edges, test_edges = train_test_split(all_edges, test_size=0.2, random_state=42)
    
    train_spatial = df[df['edge_id'].isin(train_edges)].drop(columns=['edge_id'])
    test_spatial = df[df['edge_id'].isin(test_edges)].drop(columns=['edge_id'])
    # Drop temp col from full dataset
    df.drop(columns=['edge_id'], inplace=True)
    
    train_spatial.to_csv(os.path.join(OUTPUT_DIR, "train_spatial_split.csv"), index=False)
    test_spatial.to_csv(os.path.join(OUTPUT_DIR, "test_spatial_split.csv"), index=False)
    print(f"Spatial Split: Train segments={len(train_edges)}, Test segments={len(test_edges)}")
    print(f"               Train size={len(train_spatial)}, Test size={len(test_spatial)}")
    
    print("\nFeature engineering and dataset preparation completed successfully!")

if __name__ == "__main__":
    engineer_features()
