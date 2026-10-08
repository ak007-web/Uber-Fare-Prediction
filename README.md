# Uber Fare Prediction

A small end-to-end machine learning project that estimates historical NYC Uber fares from trip coordinates, pickup time, and passenger count. It includes a reproducible training workflow, an analysis notebook, a responsive HTML/CSS/JavaScript frontend, and a local Python prediction API.

> **Project demo:** Estimates are based on historical data and are not live Uber quotes. They do not account for today's traffic, tolls, route choice, surge pricing, or Uber's current pricing rules.

## Run the app

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fare_model.py
python app.py
```

The training command reads `uber.csv` from this project directory and writes `fare_model.joblib` and `model_metrics.json`. Then start the local server with `python app.py` and open [http://127.0.0.1:8000](http://127.0.0.1:8000). The frontend accepts NYC-area pickup/drop-off coordinates, pickup date and time, and passenger count. Python serves the frontend files and exposes the model through a local JSON endpoint; the visible interface is HTML, CSS, and JavaScript.

## Project files

- `app.py` — local Python HTTP server and prediction API.
- `frontend/` — responsive HTML, CSS, and JavaScript frontend.
- `fare_model.py` — shared cleaning, feature generation, training, and prediction helpers.
- `uber.ipynb` — data review, model comparison, and error-analysis notebook.
- `uber.csv` — source dataset (200,000 records; historical NYC trips).
- `requirements.txt` — Python dependencies.

## Data and modeling

The workflow drops source index/key columns, parses UTC timestamps, removes incomplete or invalid rows, keeps positive fares up to $200 and passenger counts from 1 to 6, bounds coordinates to the NYC area, and removes near-zero or over-100 km trips. Trip distance is computed with a vectorized Haversine formula. Time features are derived in New York local time and encoded cyclically so late night/hour and year-end dates remain close in feature space.

The final 20% of trips by pickup timestamp is held out for evaluation; earlier trips are used for training. This chronological split gives a more realistic check of predictions on later trips than a random split. Metrics include MAE, RMSE, and R-squared, alongside a median-fare baseline. The notebook compares a simple linear model with gradient boosting and plots errors.

Training metrics are saved in `model_metrics.json`. The model is intended for education and demonstration, not real-world fare quotations.

## Next steps for a final year submission

Add location search or a map-based coordinate picker, more contextual inputs where available, model versioning, and a documented evaluation on newer data. Include the dataset source/license and explain data cleaning assumptions in the report and presentation.
