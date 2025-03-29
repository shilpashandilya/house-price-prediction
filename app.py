# app.py (Updated)
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and feature names
MODEL_FILE = "model/house_price_model.pkl"
FEATURES_FILE = "model/selected_features.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError("Model or features file missing. Train the model first.")

model = pickle.load(open(MODEL_FILE, "rb"))
feature_names = pickle.load(open(FEATURES_FILE, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert numeric inputs to float
        for key in ["area", "bedrooms", "bathrooms", "stories", "parking"]:
            data[key] = float(data[key])

        # Convert categorical fields
        data["furnished"] = 1 if data["furnishing"] == "furnished" else 0
        data["semi_furnished"] = 1 if data["furnishing"] == "semi-furnished" else 0
        data["air_conditioning"] = 1 if data["air_conditioning"] == "yes" else 0
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure all features exist
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing feature

        input_data = input_data[feature_names]

        # Predict price
        prediction = model.predict(input_data)
        predicted_price = float(np.expm1(prediction[0]))  # Reverse log transform

        # Convert to INR
        predicted_price_inr = predicted_price * 83  # USD to INR conversion (Example)

        return render_template("index.html", price=f"₹{predicted_price_inr:,.2f}")

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
