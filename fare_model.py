"""Training and inference utilities for the NYC Uber fare demo."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "uber.csv"
MODEL_PATH = ROOT / "fare_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
FRONTEND_PATH = ROOT / "frontend"
# Broad NYC-area bounds include the airports and surrounding pickup zones.
LAT_MIN, LAT_MAX = 40.35, 41.05
LON_MIN, LON_MAX = -74.30, -73.50
FEATURES = [
    "trip_distance_km", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "pickup_hour_sin",
    "pickup_hour_cos", "pickup_weekday_sin", "pickup_weekday_cos",
    "pickup_month_sin", "pickup_month_cos", "passenger_count",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in kilometers."""
    lat1, lon1, lat2, lon2 = [np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def valid_nyc_coordinates(lat, lon):
    return (LAT_MIN <= float(lat) <= LAT_MAX) and (LON_MIN <= float(lon) <= LON_MAX)


def make_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, pickup_datetime, passenger_count):
    dt = pd.Timestamp(pickup_datetime)
    if dt.tzinfo is None:
        dt = dt.tz_localize("America/New_York")
    # Dataset timestamps are UTC. Convert to NYC local time before deriving calendar features.
    dt = dt.tz_convert("America/New_York")
    hour, weekday, month = dt.hour + dt.minute / 60, dt.weekday(), dt.month
    return pd.DataFrame([{
        "trip_distance_km": float(haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)),
        "pickup_longitude": float(pickup_lon), "pickup_latitude": float(pickup_lat),
        "dropoff_longitude": float(dropoff_lon), "dropoff_latitude": float(dropoff_lat),
        "pickup_hour_sin": np.sin(2 * np.pi * hour / 24),
        "pickup_hour_cos": np.cos(2 * np.pi * hour / 24),
        "pickup_weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "pickup_weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "pickup_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "pickup_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "passenger_count": int(passenger_count),
    }], columns=FEATURES)


def load_and_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "passenger_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df.drop(columns=[c for c in ("Unnamed: 0", "key") if c in df.columns])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=list(required))
    for col in ["fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"])
    df = df[(df.fare_amount > 0) & (df.fare_amount <= 200) & df.passenger_count.between(1, 6)]
    for prefix in ("pickup", "dropoff"):
        df = df[df[f"{prefix}_latitude"].between(LAT_MIN, LAT_MAX) & df[f"{prefix}_longitude"].between(LON_MIN, LON_MAX)]
    df["trip_distance_km"] = haversine_km(df.pickup_latitude, df.pickup_longitude,
                                           df.dropoff_latitude, df.dropoff_longitude)
    # Drop implausibly long trips and zero-distance records; these are mostly corrupt locations.
    df = df[df.trip_distance_km.between(0.05, 100)].copy()
    local = df.pickup_datetime.dt.tz_convert("America/New_York")
    hour = local.dt.hour + local.dt.minute / 60
    weekday, month = local.dt.dayofweek, local.dt.month
    df["pickup_hour_sin"], df["pickup_hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    df["pickup_weekday_sin"], df["pickup_weekday_cos"] = np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)
    df["pickup_month_sin"], df["pickup_month_cos"] = np.sin(2*np.pi*(month-1)/12), np.cos(2*np.pi*(month-1)/12)
    return df.sort_values("pickup_datetime").reset_index(drop=True)


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = load_and_clean_data(data_path)
    if len(df) < 100:
        raise ValueError("Not enough valid trips remain after cleaning to train a model.")
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train.fare_amount, test.fare_amount
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    # Classic gradient boosting fits on the main thread, which also works in restricted
    # desktop environments where OpenMP worker process creation is unavailable.
    model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.06,
                                      max_depth=3, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model": "GradientBoostingRegressor",
        "split": "chronological 80% train / 20% test",
        "clean_rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_end_utc": train.pickup_datetime.max().isoformat(),
        "test_start_utc": test.pickup_datetime.min().isoformat(),
        "baseline_mae": round(float(mean_absolute_error(y_test, baseline.predict(X_test))), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, model_path)
    metrics_path = model_path.with_name("model_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_model(), indent=2))
