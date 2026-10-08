"""Serve the HTML frontend and a small local JSON prediction API."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

import joblib

from fare_model import MODEL_PATH, FRONTEND_PATH, make_features, valid_nyc_coordinates


class FareRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_PATH), **kwargs)

    def _json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/metrics":
            if not MODEL_PATH.exists():
                return self._json(503, {"error": "Model is missing. Run python fare_model.py first."})
            model_data = joblib.load(MODEL_PATH)
            return self._json(200, model_data["metrics"])
        if path.startswith("/api/"):
            return self._json(404, {"error": "API route not found."})
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self._json(404, {"error": "API route not found."})
        if not MODEL_PATH.exists():
            return self._json(503, {"error": "Model is missing. Run python fare_model.py first."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10_000:
                return self._json(400, {"error": "Invalid request size."})
            request = json.loads(self.rfile.read(length))
            pickup_lat = float(request["pickup_latitude"])
            pickup_lon = float(request["pickup_longitude"])
            dropoff_lat = float(request["dropoff_latitude"])
            dropoff_lon = float(request["dropoff_longitude"])
            passengers = int(request["passenger_count"])
            pickup_datetime = request["pickup_datetime"]
            if not (valid_nyc_coordinates(pickup_lat, pickup_lon) and
                    valid_nyc_coordinates(dropoff_lat, dropoff_lon)):
                return self._json(422, {"error": "Both locations must be within the supported NYC area."})
            if passengers not in range(1, 7):
                return self._json(422, {"error": "Passenger count must be between 1 and 6."})
            features = make_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                                     pickup_datetime, passengers)
            distance = float(features.iloc[0]["trip_distance_km"])
            if distance < 0.05:
                return self._json(422, {"error": "Pickup and drop-off are too close together."})
            model = joblib.load(MODEL_PATH)["model"]
            estimate = max(0.0, float(model.predict(features)[0]))
            return self._json(200, {"estimated_fare": round(estimate, 2),
                                    "distance_km": round(distance, 2)})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "Please provide valid trip details."})

    def log_message(self, fmt, *args):
        # Keep normal browser polling quiet while retaining errors in the console.
        if args and str(args[1]).startswith("4"):
            super().log_message(fmt, *args)


def main():
    if not MODEL_PATH.exists():
        print("Model artifact not found. Run `python fare_model.py` first.")
        return
    server = ThreadingHTTPServer(("127.0.0.1", 8000), FareRequestHandler)
    print("Uber Fare Prediction is running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
