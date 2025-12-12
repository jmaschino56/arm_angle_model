# Arm Angle Tester

Jupyter notebook for evaluating the arm angle prediction ML model against a geometric baseline calculation.

## Purpose

This notebook validates the trained arm angle estimator model (`models/arm_angle/arm_angle_estimator.pkl`) by comparing its predictions against:
1. **Actual arm angles** from Statcast data
2. **Geometric calculation** based on release position and pitcher height

## Data Pipeline

### Data Sources
- **Pitcher Data**: Oracle API endpoint (`/players/check/{season}`) - retrieves pitchers with arm length measurements
- **Schedule**: MLB Stats API - game schedule with game_pk identifiers
- **Play-by-Play**: Oracle API endpoint (`/patreon/GET_PBP_GAME/{game_pk}`) - pitch-level data

### Data Fetching Methods
Three optimization levels available:
| Method | Function | Speed | Use Case |
|--------|----------|-------|----------|
| Async | `get_combined_pbp_data_async()` | Fastest | Production |
| Threaded | `get_combined_pbp_data_threaded()` | Fast | Default |
| Polars | `get_combined_pbp_data_polars()` | Fast + clean | Used in notebook |

All methods include retry logic with exponential backoff (default 3 retries).

## Key Functions

### `calculate_arm_angles(p_throws, release_pos_x, release_pos_z, height)`
Geometric baseline calculation using trigonometry:
- Estimates shoulder position as 70% of height
- Calculates angle from release point using `arctan2`
- Adjusts sign for left-handed pitchers

### `model_arm_angles(df_pl, model, label_encoders, features, output_col)`
Runs the ML model on a Polars DataFrame:
- Handles categorical encoding via label encoders
- Returns predictions as NumPy array or appended column

## Evaluation Metrics

The notebook generates four visualizations:

1. **Hero Metrics Comparison** - Bar charts comparing R², MAE, RMSE, and Bias
2. **Scatter Comparison** - Predicted vs actual for both methods
3. **Error Distribution** - Histogram of prediction errors
4. **MAE by Handedness** - Performance breakdown by pitcher hand

## Results Summary

Based on 2025 season data (~2,430 games):

| Metric | ML Model | Geometric Calc | Improvement |
|--------|----------|----------------|-------------|
| R² | 0.9862 | 0.9504 | +3.8% |
| MAE (°) | 4.38 | 8.35 | **47.5% reduction** |
| RMSE (°) | 5.66 | 10.74 | 47.3% reduction |
| Bias (°) | 0.001 | -0.72 | Near zero |

## Requirements

### API Key or PyBaseball
Requires a valid API key with API Access subscription:
```python
api_key = 'your-api-key-here'
```
You can also use PyBaseball to generate the dataframe needed here.

### Dependencies
```python
# Core
import logging
import time
import pickle
import sys
import os
import random
import warnings

# Data
import numpy as np
import pandas as pd
import polars as pl
import requests
import asyncio
import aiohttp
from pytz import timezone
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# HTTP
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Database
import cx_Oracle

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

## Usage

```python
# 1. Initialize Oracle client
init_oracle_client()

# 2. Fetch data
pitchers = get_pitchers(season=2025)
schedule = get_schedule(year_input=[2025])
savant_data = get_combined_pbp_data_polars(schedule, api_key, columns_to_drop, max_workers=20)

# 3. Load model and predict
with open('models/arm_angle/arm_angle_estimator.pkl', 'rb') as f:
    bundle = pickle.load(f)

results = model_arm_angles(
    df_pl=df,
    model=bundle['model'],
    label_encoders=bundle['label_encoders'],
    features=bundle['features'],
    output_col="arm_angle_pred"
)

# 4. Run evaluation visualizations
# (see Evaluation section in notebook)
```

## Model File

**Location**: `models/arm_angle/arm_angle_estimator.pkl`

Bundle contents:
- `model`: Trained LightGBM regressor pipeline
- `label_encoders`: Dictionary of categorical encoders
- `features`: List of feature column names

## Notes

- Processing ~2,430 games takes several minutes with 20 concurrent workers
- Sample size for visualizations capped at 50,000 points for performance
- Geometric calculation assumes shoulder at 70% of height (simplified biomechanics)
