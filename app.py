from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load only XGBoost model
model = {
    "XGBoost": {
        "pipeline": joblib.load("formation_energy_pipeline.pkl"),
        "r2": 0.959
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_data = {}
    error_message = None
    
    if request.method == "POST":
        try:
            # Get input values
            input_data = {
                "energy_above_hull": float(request.form["energy_above_hull"]),
                "band_gap": float(request.form["band_gap"]),
                "energy_per_atom": float(request.form["energy_per_atom"]),
                "vbm": float(request.form["vbm"]),
                "cbm": float(request.form["cbm"])
            }
            features = np.array([[input_data[f] for f in input_data]])

            # Predict with XGBoost model
            pred = model["XGBoost"]["pipeline"].predict(features)[0]
            prediction = {
                "value": round(pred, 4),
                "r2": model["XGBoost"]["r2"]
            }
        except ValueError:
            error_message = "Invalid input. Please enter valid numbers."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render_template("index.html", 
                         prediction=prediction, 
                         input_data=input_data,
                         error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)