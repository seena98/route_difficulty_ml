import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
DATA_DIR = "data/processed"
MODEL_OUTPUT_DIR = "data"
METRICS_PATH = os.path.join(DATA_DIR, "model_evaluation_metrics.csv")
PLOT_PATH = os.path.join(DATA_DIR, "feature_importances.png")

# Set styles for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

def load_split_data(split_name):
    """Load train and test sets for a specific split strategy."""
    train_path = os.path.join(DATA_DIR, f"train_{split_name}.csv")
    test_path = os.path.join(DATA_DIR, f"test_{split_name}.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing files for split '{split_name}'. Run extractor.py first.")
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def get_features_and_targets(df, feature_cols, target_col="difficulty_score", road_only=False):
    """Separate features and targets. Optionally filter to road-only features for baseline."""
    X = df[feature_cols]
    
    if road_only:
        # Exclude driver-specific and vehicle-specific features
        exclude_prefixes = ['driver_', 'vehicle_', 'highway_comfort', 'narrow_road_comfort', 'night_vision',
                            'width_margin', 'experience_traffic', 'narrow_comfort_margin',
                            'night_vision_lit', 'experience_sinuosity', 'weight_slope']
        road_features = []
        for col in feature_cols:
            if not any(col.startswith(p) for p in exclude_prefixes):
                road_features.append(col)
        X = X[road_features]
        
    y = df[target_col]
    return X, y

def evaluate_predictions(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def train_and_evaluate_pipeline():
    print("--- Phase 5: Machine Learning Model Training & Evaluation ---")
    
    # Load feature list
    feature_list_path = os.path.join(DATA_DIR, "feature_list.txt")
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError("Missing feature_list.txt. Run extractor.py first.")
        
    with open(feature_list_path, "r") as f:
        feature_cols = [line.strip() for line in f.read().splitlines() if line.strip()]
        
    splits = ["random", "user_split", "spatial_split"]
    results = []
    
    # We will save the best model from the random split to plot importances
    best_model = None
    best_features = None
    
    for split in splits:
        print(f"\nProcessing Split: {split.upper()}...")
        train_df, test_df = load_split_data(split)
        
        # --- 1. ROAD-ONLY BASELINE MODEL (Non-Personalized) ---
        X_train_base, y_train = get_features_and_targets(train_df, feature_cols, road_only=True)
        X_test_base, y_test = get_features_and_targets(test_df, feature_cols, road_only=True)
        
        # Standard Random Forest trained on road attributes only
        base_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        base_model.fit(X_train_base, y_train)
        y_pred_base = base_model.predict(X_test_base)
        
        mae_base, rmse_base, r2_base = evaluate_predictions(y_test, y_pred_base)
        results.append({
            "Split": split,
            "Model": "Baseline (Road-Only)",
            "MAE": mae_base,
            "RMSE": rmse_base,
            "R2": r2_base
        })
        print(f"  Baseline (Road-Only) - MAE: {mae_base:.3f}, RMSE: {rmse_base:.3f}, R2: {r2_base:.3f}")
        
        # --- 2. RANDOM FOREST (Personalized) ---
        X_train_full, _ = get_features_and_targets(train_df, feature_cols, road_only=False)
        X_test_full, _ = get_features_and_targets(test_df, feature_cols, road_only=False)
        
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_full, y_train)
        y_pred_rf = rf_model.predict(X_test_full)
        
        mae_rf, rmse_rf, r2_rf = evaluate_predictions(y_test, y_pred_rf)
        results.append({
            "Split": split,
            "Model": "Random Forest (Personalized)",
            "MAE": mae_rf,
            "RMSE": rmse_rf,
            "R2": r2_rf
        })
        print(f"  Random Forest        - MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}, R2: {r2_rf:.3f}")
        
        # --- 3. XGBOOST (Personalized) ---
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_full, y_train)
        y_pred_xgb = xgb_model.predict(X_test_full)
        
        mae_xgb, rmse_xgb, r2_xgb = evaluate_predictions(y_test, y_pred_xgb)
        results.append({
            "Split": split,
            "Model": "XGBoost (Personalized)",
            "MAE": mae_xgb,
            "RMSE": rmse_xgb,
            "R2": r2_xgb
        })
        print(f"  XGBoost              - MAE: {mae_xgb:.3f}, RMSE: {rmse_xgb:.3f}, R2: {r2_xgb:.3f}")
        
        # Capture best model for plotting
        if split == "random":
            best_model = xgb_model
            best_features = X_train_full.columns
            
            # Save final trained models on Full Random split to data/ for dashboard use
            with open(os.path.join(MODEL_OUTPUT_DIR, "best_rf_model.pkl"), "wb") as file:
                pickle.dump(rf_model, file)
            with open(os.path.join(MODEL_OUTPUT_DIR, "best_xgb_model.pkl"), "wb") as file:
                pickle.dump(xgb_model, file)
            print("  [Saved trained RF & XGBoost models to data/ directory]")
            
    # Save metrics to CSV
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"\nSaved all metrics comparison to {METRICS_PATH}")
    
    # Print a markdown table for the user
    print("\nEvaluation Summary:")
    print("| Split | Model | MAE | RMSE | R² Score |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Split']} | {r['Model']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['R2']:.3f} |")
        
    # --- 4. FEATURE IMPORTANCE ANALYSIS ---
    if best_model is not None:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        top_n = 15
        sns.barplot(
            x=importances[indices[:top_n]],
            y=np.array(best_features)[indices[:top_n]],
            palette="viridis",
            hue=np.array(best_features)[indices[:top_n]],
            legend=False
        )
        plt.title(f"Top {top_n} Mapped Features for Driving Difficulty (XGBoost)")
        plt.xlabel("Relative Feature Importance")
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        plt.close()
        print(f"Saved feature importance plot to {PLOT_PATH}")

if __name__ == "__main__":
    train_and_evaluate_pipeline()
