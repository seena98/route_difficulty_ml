# 🚗 Personalized Route Difficulty Prediction Navigation System

Welcome to the **Personalized Route Difficulty Prediction Navigation System**! This software is designed for a Master's/Bachelor's thesis to demonstrate how navigation systems (like GPS or maps) can suggest different routes depending on **who** is driving and **what** they are driving.

Traditional GPS apps (like Google Maps or Apple Maps) only look at the shortest distance or the fastest time. They treat a 17-year-old novice driver in a massive motorhome (RV) exactly the same as a professional taxi driver in a small compact car. This project changes that by using **Machine Learning** (AI) to predict how difficult or stressful a route will be for a specific driver and finding a "comfort-optimized" path to detour around stressful areas.

---

## 🌟 How It Works (In Simple Terms)

1.  **Map Data**: We download the road map of **Berlin Mitte** (city center) from OpenStreetMap.
2.  **Elevation & Hills**: We add altitude data to calculate how steep every bridge or incline is.
3.  **Driver Profiles**: We define a driver by their age, driving experience, and vehicle size.
4.  **Traffic Modeling**: We simulate congestion levels based on peak rush hours (morning and evening commutes).
5.  **Personalized Cost Routing**: If you select "Personalized Driver Comfort", the map doesn't just calculate the shortest distance; it multiplies the distance of a road by its predicted difficulty. The system then automatically plans a route that detours around streets that are too narrow, too winding, unlit at night, or highly congested.

---

## 👤 The Driver Profile (Personalization Inputs)

To personalize the difficulty, the system looks at:
*   **Driver Age**: Used to simulate night-vision capability (which naturally decreases as we get older, making unlit streets more difficult at night).
*   **Driving Experience (Years)**: Novice drivers (less than 2 years of experience) are simulated to experience higher stress during heavy traffic congestion, large roundabouts, and complex multi-lane highway merges.
*   **Vehicle Category**:
    *   *Compact Car* (narrow, light; sensitive to bumpy cobblestone roads).
    *   *Sedan* (average size/weight).
    *   *SUV* (wide, heavy; experiences stress on narrow streets).
    *   *RV / Caravan* (extremely wide and heavy; experiences high stress on narrow streets, two-way corridors, and steep slopes).

---

## 📊 The 30 Route Features (Explained)

We look at **30 different characteristics** of each street segment to evaluate its difficulty. Here is what they mean in plain English:

### A. Road Geometry (The shape of the street)
1.  **Average Gradient**: How steep the road is on average (in %).
2.  **Max Gradient**: The steepest climb or descent on that specific street segment.
3.  **Sinuosity (Windingness)**: A ratio showing how curved a road is. A straight road has a value of 1.0. A winding road with lots of curves will have a value of 1.5 or higher.
4.  **Curve Density**: How many sharp turns (turns of 30 degrees or more) there are per kilometer.
5.  **Lane Count**: The number of lanes available for driving on this segment.
6.  **Road Width**: The physical width of the drivable road in meters.
7.  **Has Median Divider**: Whether oncoming traffic is separated by a physical barrier (like concrete blocks or grass) which makes driving less stressful.
8.  **Is One-Way**: Whether the street is a one-way street (two-way narrow streets are much harder because you have to squeeze past oncoming cars).

### B. Infrastructure & Quality (The street's condition)
9.  **Road Class**: The importance of the road (e.g., Motorway/Highway, Main Road, Secondary Road, Neighborhood/Residential Road).
10. **Surface Type**: What the road is made of (Asphalt, Cobblestones, Gravel, or Dirt).
11. **Smoothness Index**: The quality of the pavement (how smooth it is vs. potholes and rough surfaces).
12. **Is Lit**: Whether the street has streetlights (critical for visibility at night).
13. **Has Tunnel**: Whether the street goes through a tunnel (GPS loss, dim lighting, and narrow boundaries increase stress).
14. **Has Bridge**: Whether the street crosses a bridge (high winds, no shoulders).

### C. Junctions & Traffic Control (Friction points)
15. **Junction Density**: How many intersections or cross streets there are per kilometer.
16. **Has Roundabout**: Whether the road leads into or contains a roundabout (which requires yielding and rapid lane decisions).
17. **Stop Sign Count**: How many stop signs or yield signs are along the street.
18. **Traffic Light Count**: How many stoplights are on this road segment.
19. **Has Speed Bump**: Presence of speed bumps or chicanes designed to slow down vehicles.

### D. Road Hazards & Other Users
20. **Has Tram Tracks**: Whether tram rails are embedded directly in the street surface (slippery when wet, requiring extra tires-crossing care).
21. **Has Cycleway**: Whether there is a bicycle lane right next to the car lane (requires constant blind-spot checking when turning).
22. **Pedestrian Crossing Density**: The frequency of crosswalks (requiring frequent stops and high vigilance).
23. **Has On-Street Parking**: Presence of parked cars along the curb (risk of doors opening, narrowing the drivable space).

### E. Surrounding Environment (Where you are driving)
24. **Is Urban**: Driving inside a dense city center (lots of buildings, pedestrians, and close hazards).
25. **Is Forest / Jungle**: Driving in a dense park, forest, or wooded area (darker, leaves on the road, risk of wild animals crossing).
26. **Is Mountainous**: High elevation shifts, winding cliffs, or lack of guardrails.
27. **Is Rural Open**: Country roads in open fields (windy, high speeds).

### F. Traffic & Operations
28. **Speed Limit**: The legal maximum speed limit in km/h.
29. **Traffic Congestion**: The level of traffic (from 0.0 - free flow, to 1.0 - bumper-to-bumper gridlock) depending on the time of day.
30. **Lane Change Density**: How often you need to merge or switch lanes (high merge stress).

---

## 📖 Plain-English Glossary of Technical Terms

If you do not have a technical background in AI or mapping systems, here are simple explanations for the terms used in this project:

### Mapping & Geodata Terms
*   **Node**: An intersection, junction, or endpoint on a map (represented by latitude and longitude coordinates).
*   **Edge**: A single street segment that connects two nodes (intersections).
*   **Directed Graph (Road Network)**: A math term for a map where streets can have direction (like one-way streets). In our system, Berlin is represented as a directed graph.
*   **GraphML**: A standard text-based file format used to save map networks so that computers can load them instantly without redownloading them from the web.
*   **Geocoding**: The process of taking a text address (like `"Schloss Charlottenburg"`) and looking up its exact GPS coordinates so the computer can plot it.
*   **Nominatim API**: A free public search engine run by OpenStreetMap that performs geocoding lookups.
*   **SRTM (Shuttle Radar Topography Mission)**: A global database of satellite-measured altitudes. We use it offline to find out how high each street is and calculate hills.

### Machine Learning (AI) Terms
*   **Machine Learning (ML)**: A type of AI where the computer looks at historical data (simulated driving logs) to find patterns and learn how to make predictions (route stress scores) on its own, rather than using fixed rules.
*   **Random Forest**: An AI algorithm that works by creating hundreds of "decision trees" (like flowcharts). It asks questions about the road (width, slope) and the driver (experience) and averages their answers to predict stress.
*   **XGBoost (Extreme Gradient Boosting)**: A high-performance AI algorithm. It works similarly to a Random Forest but builds decision trees one after another, with each new tree trying to fix the mistakes made by the previous ones.
*   **Heuristic Baseline**: A simple, non-personalized rule-of-thumb model used for comparison. For example, our baseline simply assumes: *"steeper hills = harder road"* for everyone, ignoring the driver's age or vehicle width.
*   **One-Hot Encoding**: A method used to translate words (like `"Compact"`, `"SUV"`, `"RV"`) into columns of `1`s and `0`s so that machine learning algorithms can calculate them.
*   **Interaction Features / Terms**: Combining two inputs to help the AI learn complex rules. For example, multiplying `driver_age` by `is_night` tells the AI to look at age *specifically* when it is dark outside.

### Evaluation Metrics (How we measure accuracy)
*   **MAE (Mean Absolute Error)**: The average size of the model's prediction mistakes. A score of `0.22` means if the driver's true stress rating is `3.0`, the AI is off by an average of `0.22` (predicting around `2.78` or `3.22`). **Lower is better.**
*   **RMSE (Root Mean Squared Error)**: Similar to MAE, but it penalizes larger mistakes much more heavily. A low RMSE means the model rarely makes huge errors. **Lower is better.**
*   **R² Score (R-Squared)**: A percentage showing how much of the driver's stress variation is successfully explained by our model. A score of `0.91` (or `91%`) means our AI successfully explains 91% of why different drivers experience different stress levels, leaving only 9% to random chance. **Higher is better.**
*   **Data Split Strategies (Random, User, Spatial)**: Ways to test if the AI is cheating or actually learning:
    *   *Random Split*: We hide a random 20% of the driving logs, train on the remaining 80%, and see if the AI can predict the hidden ones.
    *   *User Split*: We train the AI on 160 drivers and test it on 40 completely new drivers. This proves the AI generalized to *new people* it has never seen before.
    *   *Spatial Split*: We train the AI on 80% of Berlin's streets and test it on 20% of roads in neighborhoods it has never visited. This proves the AI generalized to *new places*.

---

## 🛠️ Interactive Feature Sensitivities (Sidebar Sliders)

In the dashboard sidebar under **Custom Feature Sensitivities**, you can manually adjust sliders from `0.0` (ignore) to `3.0` (highly sensitive) to customize what makes driving hard:
*   *If set to 1.0*: The system uses the default AI machine learning predictions.
*   *If set to 3.0 (e.g., Narrow Roads)*: It multiplies the difficulty of narrow roads. If you select "Personalized Comfort" routing, the map will immediately detour you to a wider road.
*   *If set to 0.0 (e.g., Traffic Congestion)*: It tells the routing engine that you do not mind heavy traffic, causing it to ignore congestion stress.

---

## 🛣️ Route Selection & Driver Choice Dashboard

When you select a start and end point in Berlin, the system calculates and displays **three distinct routes side-by-side**:

1.  **Option 1: Shortest Path**: The standard routing that minimizes physical travel distance. This route often routes through narrow or congested segments because it ignores driver comfort, frequently leading to a **Hard/Difficult** or **Moderate Stress** rating.
2.  **Option 2: Comfort-Optimized Path**: Sourced directly from our personalized machine learning models. By minimizing the personalized difficulty weight of each segment, it automatically routes around stressful bottlenecks, resulting in an **Easy Comfort** or **Moderate Stress** rating.
3.  **Option 3: Alternative Bypass**: A third route option calculated by penalizing the segments of Options 1 & 2. This forces the pathfinder onto physically distinct roads to offer a clear alternative bypass.

### 📍 Start & End Point Input Modes:
Rather than being limited to a static list, you can select locations using four flexible input methods:
*   **📍 Preset Landmarks**: Choose from 8 major landmarks in Berlin (e.g., Brandenburg Gate, Alexanderplatz, Hauptbahnhof) for quick testing.
*   **🔍 Custom Address Search**: Type any location, address, or intersection name (e.g., `"Kurfürstendamm 100"`, `"Schloss Bellevue"`). The dashboard uses OpenStreetMap's Nominatim API to geocode the query and snap to the nearest road network node.
*   **🌐 Coordinate Input**: Enter custom latitude and longitude coordinates directly (e.g. `52.5162`, `13.3777`).
*   **🎲 Random Graph Nodes**: Automatically choose random node intersections from the loaded road network graph—highly useful for exploring the entire city map.

### Interactive Decision Features:
*   **Difficulty Index Labels**: Every route option is assigned a clear overall difficulty rating:
    *   🟢 **Easy Comfort** (Average stress score < 2.0 / 5.0)
    *   🟡 **Moderate Stress** (Average stress score between 2.0 and 3.2 / 5.0)
    *   🔴 **Hard/Difficult** (Average stress score > 3.2 / 5.0)
*   **Driver Choice Confirmation**: Click the **Accept and Launch Navigation** button to confirm your selection and trigger a simulated GPS link to your vehicle dashboard.
*   **Street-by-Street Hazard Breakdown**: Below the map, expand the table to see a detailed analysis of every street segment on the selected route. It shows the length, slope, road width, surface type, and predicted difficulty index, along with a list of specific stress factors (e.g., "Narrow for vehicle width", "Steep slope", "Rough cobblestones").

---


## 📂 Project Structure

```
routeDifficulty/
├── data/
│   ├── elevation_cache.sqlite    # SQLite database caching node elevations
│   ├── berlin_mitte_drive.graphml # The downloaded road network file
│   ├── driver_profiles.csv        # 200 simulated driver profiles
│   ├── simulated_driving_logs.csv # 170,000+ segment logs used for ML training
│   └── processed/
│       ├── full_processed_dataset.csv # Encoded dataset
│       ├── train_random.csv / test_random.csv   # standard 80/20 split
│       ├── train_user_split.csv ...             # split for unseen drivers
│       └── train_spatial_split.csv ...          # split for unseen roads
├── src/
│   ├── data_collection/
│   │   ├── verify_osmnx.py        # Small script to test OSM downloader
│   │   └── downloader.py          # Main map and elevation download pipeline
│   ├── simulation/
│   │   └── driver_simulator.py    # Generates profiles and simulates driving logs
│   ├── features/
│   │   └── extractor.py           # Feature engineering and split generation
│   ├── models/
│   │   └── train.py               # Model training (RF & XGBoost) and evaluation
│   └── app/
│       └── main.py                # Streamlit navigation dashboard (Web App)
├── thesis/
│   └── structure.md               # Latex templates, metrics, and chapter outline
├── requirements.txt               # Required Python packages
└── README.md                      # This document
```

---

## 🚀 How to Run the Project

## 🚀 Detailed Step-by-Step Pipeline Execution Guide

To run the project or reproduce its results, follow this detailed guide. We execute the pipeline sequentially in six distinct phases.

---

### Phase 1: Environment & Virtual Setup

Before running the code, set up the Python virtual environment and install dependencies.

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   ```
2. **Activate the Environment**:
   * On macOS / Linux:
     ```bash
     source venv/bin/activate
     ```
   * On Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   * *What gets installed*: `osmnx` (map data), `geopandas` (spatial dataframes), `srtm.py` (offline elevation data), `xgboost` (gradient boosting), `scikit-learn` (machine learning utilities), `streamlit` (dashboard app), and `folium` (interactive leaflet maps).

---

### Phase 2: Map Data Collection & Slope Calculation

In this phase, we fetch the physical road network of Berlin and integrate altitude elevation data to calculate road inclines (slopes).

1. **Run Map Downloader Command**:
   ```bash
   python src/data_collection/downloader.py
   ```
2. **How it works under the hood**:
   * **Graph Download**: The script queries OpenStreetMap using `osmnx` to download all drivable roads in Berlin, Germany. It yields a NetworkX directed graph containing node coordinates (latitude/longitude) and edge attributes (road type, lanes, speed limits, surface, one-way markers).
   * **Elevation Integration**: To calculate slope gradients, it uses the offline Python `srtm` package to fetch elevation tiles. It queries the elevation (meters above sea level) for all 28,432 nodes in our Berlin graph.
   * **Caching**: Checked and newly queried coordinates are saved to a local SQLite database at `data/elevation_cache.sqlite` to prevent duplicate fetches.
   * **Slope Calculation**: For every directed street segment (edge) from node $u$ to node $v$, it computes the slope percentage using:
     $$\text{Slope (\%)} = \left(\frac{\text{Elevation}_v - \text{Elevation}_u}{\text{Edge Length}}\right) \times 100$$
   * **Save Output**: The preprocessed graph is saved in GraphML format at [berlin_mitte_drive.graphml](file:///Users/seena/VSCodePoject/routeDifficulty/data/berlin_mitte_drive.graphml).

---

### Phase 3: Traffic & Driver Commuting Simulation

Because there is no open database of personalized driver stress levels, we run a heavy synthetic driving log simulator based on human behavioral research.

1. **Run Driver Simulation Command**:
   ```bash
   python src/simulation/driver_simulator.py
   ```
2. **How it works under the hood**:
   * **Driver Profiles**: It generates 200 driver profiles containing variables for age, experience (years), vehicle category, and baseline comfort sensitivities. Saved to [driver_profiles.csv](file:///Users/seena/VSCodePoject/routeDifficulty/data/driver_profiles.csv).
   * **Commute Simulation**: It simulates **40 trips per driver** (8,000 total trips across Berlin). For each trip, it picks a random start and end intersection in the graph and routes them using the shortest distance.
   * **Dynamic Traffic Congestion**: To simulate realistic commutes, traffic levels are dynamically calculated using diurnal curves based on peak rush hours (8 AM morning and 5 PM evening commutes).
   * **Personalized Stress Formula**: As the driver traverses each segment, the simulator calculates a subjective stress rating ($1.0 - 5.0$) based on interactions between the driver's profile and the road characteristics. For example:
     * *Narrow roads* (difference between road width and vehicle width) stress wide SUVs or RVs.
     * *Cobblestone streets* stress lightweight compact cars.
     * *Unlit streets at night* stress older drivers with reduced night vision.
     * *Heavy traffic merges* stress inexperienced drivers (less than 2 years experience).
   * **Save Output**: The simulator writes **898,656 segment-level driving logs** to [simulated_driving_logs.csv](file:///Users/seena/VSCodePoject/routeDifficulty/data/simulated_driving_logs.csv).

---

### Phase 4: Feature Engineering & Dataset Preparation

This step prepares the raw driving logs for training by encoding attributes and engineering interaction terms.

1. **Run Feature Extraction Command**:
   ```bash
   python src/features/extractor.py
   ```
2. **How it works under the hood**:
   * **Categorical One-Hot Encoding**: Encodes text variables like road classification (`primary`, `residential`, etc.), vehicle category (`RV`, `Sedan`, etc.), and surface types (`asphalt`, `cobblestone`).
   * **Cross-Interaction Features**: Constructs interaction features to help the model learn the personalization rules:
     * `width_margin` = road width - vehicle width
     * `experience_traffic` = driver experience $\times$ traffic factor
     * `narrow_comfort_margin` = driver comfort sensitivity $\times$ width margin
     * `night_vision_lit` = driver night vision $\times$ lit status $\times$ night flag
     * `weight_slope` = vehicle weight $\times$ road slope
   * **Data Splitting Strategy**: Generates three validation splits saved in `data/processed/` to strictly test model generalization:
     * **Random Split**: Traditional 80/20 split on all records.
     * **User-based Split**: Trains on 160 drivers and tests on 40 completely unseen drivers (proves generalization to new drivers).
     * **Spatial Split**: Trains on 80% of road segments and tests on 20% of completely unseen roads (proves generalization to new neighborhoods).

---

### Phase 5: Machine Learning Model Training & Evaluation

Here, we train the models, compare them against a baseline, and output the finalized weights.

1. **Run Model Training Command**:
   ```bash
   python src/models/train.py
   ```
2. **How it works under the hood**:
   * **Models Trained**: Fits a **Non-Personalized Heuristic Baseline** (using average gradients only), a **Random Forest Regressor** (100 estimators, max depth 16), and an **XGBoost Regressor** (100 estimators, max depth 7).
   * **Evaluation**: Evaluates predictions on the three split test sets, computing Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
   * **Save Outputs**:
     * Exported model binaries: [best_rf_model.pkl](file:///Users/seena/VSCodePoject/routeDifficulty/data/best_rf_model.pkl) and [best_xgb_model.pkl](file:///Users/seena/VSCodePoject/routeDifficulty/data/best_xgb_model.pkl).
     * Metric comparisons: `data/processed/model_evaluation_metrics.csv`.
     * Feature importance visualization: `data/processed/feature_importances.png`.

---

### Phase 6: Launching the Interactive Web Dashboard

Deploy the Streamlit web server to visualize the personalized routes on a Folium leaflet map.

1. **Run Dashboard Command**:
   ```bash
   streamlit run src/app/main.py
   ```
2. **Interacting with the App**:
   * Open the browser and visit **`http://localhost:8501`**.
   * Configure the driver age, experience, and vehicle in the sidebar.
   * Enter a custom search query (e.g. `"Kurfürstendamm 100"`) or select presets to set start/end locations.
   * Compare the Shortest, Comfort-Optimized, and Alternative Bypass paths side-by-side.
   * Review model accuracy metrics and feature importances at the bottom of the page!

