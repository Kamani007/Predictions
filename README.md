# Solar Cell Degradation Prediction Project

## Overview
Machine learning pipeline for predicting solar cell degradation patterns and time-to-failure using pixel-level device performance data.

## Project Structure
```
Predictions/
├── artificial_2.csv                          # Pixel-level performance dataset
├── generate_all_batch_pixel_level.py         # Data preprocessing script
├── predictive_model.ipynb                    # ML pipeline notebook
└── README.md                                 # This file
```

## Data Description

### Input Dataset (`artificial_2.csv`)
- **162,132 measurements** from solar cell devices
- **Pixel-level data**: 4 pixels per device (a, b, c, d)
- **Batches**: 8 batches (45, 53, 58, 59, 65, 68, 69, 72)
- **43 unique device-batch combinations**

**Key Features:**
- `PCE_F/R`: Power Conversion Efficiency (Forward/Reverse scan)
- `FF_F/R`: Fill Factor
- `J_sc_F/R`: Short-circuit current density (mA/cm²)
- `V_oc_F/R`: Open-circuit voltage (V)
- `Max_Power_F/R`: Maximum power output (mW/cm²)
- `R_shunt_F/R`: Shunt resistance (Ω·cm²)
- `R_series_F/R`: Series resistance (Ω·cm²)
- `HI_F/R`: Hysteresis Index (%)
- `Stack`: Material composition identifier
- `Station`: Testing equipment (LS, Sunbrick, EC)

## Data Processing Pipeline

### 1. Pixel-Level Preprocessing (`generate_all_batch_pixel_level.py`)

**Key Functions:**
- **Swap Detection**: Automatically detects and corrects mislabeled Forward/Reverse measurements
- **Batch Processing**: Handles 8 batches with device-specific preprocessing
- **Dynamic Burn-in**: Identifies stabilization period per device
- **Special Cases**: Combines data from multiple test dates (e.g., Batch 65)

**Input:** Raw lifetime CSV files from Synology storage  
**Output:** `all_batch_new.csv` with pixel-level preprocessed data

**Processing Logic:**
1. Load device-stack-station mapping from `runs.csv`
2. Find largest/most recent lifetime files (prioritizes `*_new.csv`)
3. Analyze reference patterns (F vs R typically higher)
4. Detect and swap incorrectly labeled measurements
5. Output pixel-level data preserving all 4 pixels

### 2. Feature Engineering (`predictive_model.ipynb`)

#### Phase 1: Data Loading & Averaging
- Loads pixel-level data
- Averages Forward/Reverse measurements
- Generates Stack-Station distribution visualization

#### Phase 2: Pixel Filtering & Health Metrics
**Strict Filtering Logic:**
- ✅ **GOOD Pixels**: Initial dip → Recovery to peak → Gradual degradation
- ❌ **BAD Pixels**: Sharp drop with no recovery, monotonic decline, early failure

**Filtering Rules:**
- 1 bad pixel → Remove it, use remaining 3
- 2 bad pixels → Remove them, use remaining 2  
- 3+ bad pixels → Remove entire device

**Aggregated Features:**
1. **Pixel Degradation Gap (PDG)**: Difference between worst and best pixel
2. **Pixel Volatility**: Maximum fluctuation across time windows
3. **Failing Pixel Count**: Number of pixels below 80% of mean
4. **Pixel Synchronization**: Mean pairwise correlation of pixel trajectories

#### Phase 3: Temporal Feature Extraction
**Burn-in Period:** Fixed 3-hour cutoff for all devices

**Extracted Features:**
- `Peak_PCE`: Maximum efficiency after burn-in
- `Time_to_Peak`: Hours to reach peak (after burn-in)
- `Early_Decline_Rate`: Degradation rate (peak to +30h)
- `Late_Decline_Rate`: Degradation rate (after +30h)
- `Changepoint_Time`: Major behavioral shift detection
- `Reached_T80`: Boolean (reached 80% of peak)
- `Time_to_T80`: Hours after peak to reach T80 threshold
- `Total_Degradation_%`: Percentage decline from peak

#### Phase 4: Multi-Scale Pattern Decomposition
**Three Time Scales:**
- **Short-term (3h)**: Captures rapid fluctuations (2.0% volatility threshold)
- **Medium-term (10h)**: Baseline analysis (1.5% threshold)
- **Long-term (20h)**: Overall trend (1.0% threshold)

**Two-Dimensional Classification:**

**DIMENSION 1 - Primary Pattern (slope-based):**
- **Sharp**: Rapid decline (slope < -0.1% per hour)
- **Steady**: Gradual decline (moderate slope)
- **Stable**: Minimal change (|slope| < 0.02% per hour)

**DIMENSION 2 - Fluctuation (independent):**
- Detrended volatility analysis
- Point-based deviation counting
- Segment-based timeline reconstruction

**Output Features per Scale:**
- `Sharp_{scale}_%`: Percentage of time in sharp decline
- `Steady_{scale}_%`: Percentage of time in steady decline
- `Stable_{scale}_%`: Percentage of time stable
- `Fluctuating_{scale}_%`: Percentage of fluctuating points
- `Avg_Volatility_{scale}`: Mean detrended volatility
- `Max_Volatility_{scale}`: Peak volatility value

#### Phase 5: Change Point Detection
**Methods:**
1. **Pattern Transitions**: When degradation pattern changes (Sharp → Steady → Stable)
2. **Fluctuation Transitions**: When volatility starts/stops (independent)
3. **Slope Changepoints**: Significant degradation rate shifts

## Key Architecture Decision

**Learning Strategy: Learn from Pixels, Predict for Devices**

- **Training Data**: Pixel-level features (4 pixels × 43 devices)
- **Prediction Target**: Device-level outcomes
- **Rationale**: Pixels reveal HOW degradation happens, but device performance matters

## Machine Learning Models

### 1. Classification (Degradation Pattern)
**Input:** Peak_PCE, Time_to_Peak, Early_Decline_Rate, Avg_PDG  
**Output:** ['Sharp', 'Steady', 'Fluctuating', 'Stable']  
**Use:** Identify failure mode early

### 2. Survival Analysis (Time-to-Failure)
**Input:** All extracted features  
**Output:** P(survives beyond time t)  
**Use:** Predict warranty period

### 3. Root Cause Analysis
**Pattern Detection:**
- High PDG + Early decline → Manufacturing defect
- Low PDG + Late decline → Material aging
- R_shunt change → Insulation failure
- R_series change → Contact degradation

## Expected Outcomes

1. **Feature Dataset**: Device-batch rows with pixel health metrics
2. **Degradation Classes**: Behavioral pattern labels per device-batch
3. **T80 Predictions**: Time to 20% degradation threshold
4. **Performance Curves**: Predicted vs actual PCE trajectories
5. **Feature Importance**: Which pixel patterns predict device failure

## Requirements

### Python Packages
```python
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
lifelines>=0.27.0
xgboost>=1.5.0
```

### Data Sources
- **Synology Path**: `C:\Users\MahekKamani\SynologyDrive\Rayleigh\Lab Data\Sunbrick station\device data\4-stability - data\AF study\AF test`
- **Workspace**: `c:\Users\MahekKamani\OneDrive - Rayleigh Solar Tech Inc\Desktop\AF_analysis`
- **runs.csv**: Device-Stack-Station mapping file

## Usage

### 1. Generate Preprocessed Dataset
```python
python generate_all_batch_pixel_level.py
```
**Output:** `all_batch_new.csv` in workspace directory

### 2. Run ML Pipeline
Open `predictive_model.ipynb` in Jupyter and execute cells sequentially:

**Phase 1:** Load and visualize data  
**Phase 2:** Filter pixels and extract health metrics  
**Phase 3:** Calculate temporal features  
**Phase 4:** Decompose patterns at multiple scales  
**Phase 5:** Detect behavioral transitions  

### 3. Query Specific Device
```python
QUERY_DEVICE_ID = 'S003-A4_NM'
QUERY_BATCH = 58
# Filter and display device temporal features
```

## Key Insights

### Data Quality
- **Duplicate Handling**: Averages duplicate time-pixel entries
- **Anomalous Pixels**: Filters out pixels with non-declining trajectories
- **Missing Data**: Skips devices with <10 timestamps or insufficient decline

### Device-Batch Relationships
Some devices appear in multiple batches with different test conditions. Each device-batch combination is treated as a **separate data point** (different physical devices with same naming convention).

### Validation Approach
Device-level aggregated values should match `Batch_Analysis/batch XX/results.csv` files exactly.

## File Specifications

### `all_batch_new.csv` Columns
```
Batch, Device_ID, Sheet_ID, Pixel_ID, Station, Stack, Date, Time_hrs,
PCE_F, PCE_R, FF_F, FF_R, J_sc_F, J_sc_R, V_oc_F, V_oc_R,
Max_Power_F, Max_Power_R, R_shunt_F, R_shunt_R, R_series_F, R_series_R,
HI_F, HI_R
```

### `df_temporal` DataFrame Columns
```
Device_ID, Batch, Peak_PCE, Time_to_Peak, Burn_in_Time,
Early_Decline_Rate, Late_Decline_Rate, Changepoint_Time,
Reached_T80, Time_to_T80, Total_Degradation_%
```

### `df_pattern_decomposition` Columns (per scale)
```
Device_ID, Batch,
Sharp_{scale}_%, Steady_{scale}_%, Stable_{scale}_%,
Fluctuating_{scale}_%, Avg_Volatility_{scale}, Max_Volatility_{scale},
N_Windows_{scale}
```

## Configuration

### User-Configurable Parameters
```python
# In predictive_model.ipynb

# Filtering display
SHOW_DEVICE_TEMPORAL = 'S003-A4_NM'  # Device to display (empty = all)
SHOW_BATCH_TEMPORAL = 58             # Batch number

# Pattern analysis
SHOW_DEVICE_ID = 'S003-A3-SLOPE-10'  # Device for detailed output
SHOW_BATCH = 58                      # Batch for detailed output

# Window configurations
WINDOW_CONFIGS = [
    {'size': 3, 'overlap': 0.5, 'volatility_threshold': 0.02, 'name': 'short_term'},
    {'size': 10, 'overlap': 0.5, 'volatility_threshold': 0.015, 'name': 'medium_term'},
    {'size': 20, 'overlap': 0.5, 'volatility_threshold': 0.01, 'name': 'long_term'}
]
```

## Important Notes

1. **Burn-in Period**: Fixed 3-hour cutoff applied to all devices before peak detection
2. **Peak Detection**: Maximum PCE after 3-hour burn-in period
3. **Time References**: All temporal metrics (Time_to_T80, degradation rates) are measured **after peak**, not from T0
4. **Pattern Independence**: Pattern (Sharp/Steady/Stable) and Fluctuation are independent dimensions
5. **Segment-Based Calculation**: Overall fluctuation uses point-based counting within pattern segments for consistency

## Troubleshooting

### Common Issues
- **Insufficient Data**: Devices need ≥10 timestamps post-burn-in
- **No Clear Decline**: Devices must show ≥7% drop from initial to final PCE
- **Excessive Fluctuation**: Devices with >35% direction changes are filtered out
- **Anomalous Pixels**: Majority bad pixels (3+) cause device exclusion

### Skipped Device Reasons
1. Removed 3+ bad pixels (majority anomalous)
2. Insufficient timestamps (<10)
3. No clear decline (Δ<7%)
4. Excessive fluctuation (>35% direction changes)
5. Noisy trajectory (volatility >12% of mean)
6. Erratic post-peak behavior (<55% declining after peak)

## Authors
Developed at Rayleigh Solar Tech Inc.

## License
Internal use only - Proprietary
