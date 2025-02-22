from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
import requests

# Load trained LSTM models
flood_model = load_model("flood_lstm_model.h5")
earthquake_model = load_model("earthquake_lstm_model.h5")

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

# Function to get real-time weather data
def get_weather_data(lat, lon):
    api_key = "476100039f544a1e856102658250102"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}"
    response = requests.get(url)
    weather = response.json()
    return {
        "temperature": weather["current"]["temp_c"],
        "humidity": weather["current"]["humidity"],
        "rainfall": weather["current"]["precip_mm"],
        "wind_speed": weather["current"]["wind_kph"]
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/flood")
def flood_page():
    return render_template("flood.html")

@app.route("/earthquake")
def earthquake_page():
    return render_template("earthquake.html")

@app.route("/predict_flood", methods=["POST"])
def predict_flood():
    try:
        data = request.get_json()
        year, month, day = int(data["year"]), int(data["month"]), int(data["day"])
        duration, lat, lon = float(data["duration"]), float(data["latitude"]), float(data["longitude"])
        
        weather_data = get_weather_data(lat, lon)
        input_sequence = np.array([[year, month, day, duration, lat, lon,
                                    weather_data["rainfall"], weather_data["humidity"],
                                    weather_data["temperature"], weather_data["wind_speed"]]] * 10)
        input_sequence = input_sequence.reshape(1, 10, 10)
        predicted_severity = int(flood_model.predict(input_sequence)[0][0])
        
        return jsonify({
            "latitude": lat,
            "longitude": lon,
            "date": f"{year}-{month}-{day}",
            "predicted_flood_severity": predicted_severity,
            "real_time_weather": weather_data
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict_earthquake", methods=["POST"])
def predict_earthquake():
    try:
        data = request.get_json()
        lat, lon = float(data["latitude"]), float(data["longitude"])
        depth, year, month, day = float(data.get("depth", 10)), int(data["year"]), int(data["month"]), int(data["day"])
        
        input_sequence = np.array([[lat, lon, depth, year, month, day]] * 10).reshape(1, 10, 6)
        predicted_mag = float(earthquake_model.predict(input_sequence)[0][0])
        
        risk_level = "Extreme (Disaster)" if predicted_mag >= 8 else (
            "Very Severe" if predicted_mag >= 7 else (
                "Severe" if predicted_mag >= 6 else (
                    "High" if predicted_mag >= 5 else (
                        "Moderate" if predicted_mag >= 4 else (
                            "Low" if predicted_mag >= 3 else "Very Low"
                        )
                    )
                )
            )
        )
        
        return jsonify({
            "latitude": lat,
            "longitude": lon,
            "date": f"{year}-{month}-{day}",
            "predicted_magnitude": round(predicted_mag, 2),
            "risk_level": risk_level
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)