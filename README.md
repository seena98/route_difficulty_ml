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

## 🛠️ Interactive Feature Sensitivities (Sidebar Sliders)

In the dashboard sidebar under **Custom Feature Sensitivities**, you can manually adjust sliders from `0.0` (ignore) to `3.0` (highly sensitive) to customize what makes driving hard:
*   *If set to 1.0*: The system uses the default AI machine learning predictions.
*   *If set to 3.0 (e.g., Narrow Roads)*: It multiplies the difficulty of narrow roads. If you select "Personalized Comfort" routing, the map will immediately detour you to a wider road.
*   *If set to 0.0 (e.g., Traffic Congestion)*: It tells the routing engine that you do not mind heavy traffic, causing it to ignore congestion stress.

---

## 🛣️ Route Selection & Driver Choice Dashboard

When you select a start and end point in Berlin Mitte, the system calculates and displays **three distinct routes side-by-side**:

1.  **Option 1: Shortest Path**: The standard routing that minimizes physical travel distance. This route often routes through narrow or congested segments because it ignores driver comfort, frequently leading to a **Hard/Difficult** or **Moderate Stress** rating.
2.  **Option 2: Comfort-Optimized Path**: Sourced directly from our personalized machine learning models. By minimizing the personalized difficulty weight of each segment, it automatically routes around stressful bottlenecks, resulting in an **Easy Comfort** or **Moderate Stress** rating.
3.  **Option 3: Alternative Bypass**: A third route option calculated by penalizing the segments of Options 1 & 2. This forces the pathfinder onto physically distinct roads to offer a clear alternative bypass.

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

1.  **Set up the Virtual Environment**:
    ```bash
    # Create the environment
    python3 -m venv venv
    
    # Activate the environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Download Map Data (Phase 2)**:
    ```bash
    python src/data_collection/downloader.py
    ```

3.  **Run Simulation & Logs Generation (Phase 3)**:
    ```bash
    python src/simulation/driver_simulator.py
    ```

4.  **Extract Features & Create Splits (Phase 4)**:
    ```bash
    python src/features/extractor.py
    ```

5.  **Train ML Models (Phase 5)**:
    ```bash
    python src/models/train.py
    ```

6.  **Launch the Interactive Dashboard (Phase 6)**:
    ```bash
    streamlit run src/app/main.py
    ```
    Once launched, open your web browser and navigate to **`http://localhost:8501`** to interact with the map!
