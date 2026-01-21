from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Read input values from form
        radius = float(request.form["radius"])
        texture = float(request.form["texture"])
        perimeter = float(request.form["perimeter"])
        area = float(request.form["area"])
        smoothness = float(request.form["smoothness"])

        # Convert input to numpy array
        input_data = np.array([[radius, texture, perimeter, area, smoothness]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        result = model.predict(input_scaled)[0]

        if result == 1:
            prediction = "Benign"
        else:
            prediction = "Malignant"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)