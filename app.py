from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 📌 Load the trained model & scaler
model = joblib.load("soil_moisture_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 📌 Get input data
        input_data = [float(request.form[key]) for key in request.form]
        input_data = np.array(input_data).reshape(1, -1)

        # 📌 Scale input
        input_data_scaled = scaler.transform(input_data)

        # 📌 Predict Soil Moisture
        prediction = model.predict(input_data_scaled)[0]

        return jsonify({"soil_moisture": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
