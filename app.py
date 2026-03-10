from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize Flask
app = Flask(__name__)

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return "Crop Recommendation API is Running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction = model.predict(input_features)

        result = prediction[0]

        return jsonify({
            "success": True,
            "recommended_crop": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)