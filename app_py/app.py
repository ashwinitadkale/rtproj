from flask import Flask, render_template, request
import pickle
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        cycle_length = float(request.form.get("cycle_length"))
        period_length = float(request.form.get("period_length"))
        stress_level = float(request.form.get("stress_level"))
        exercise_frequency = float(request.form.get("exercise_frequency"))
        diet_quality = float(request.form.get("diet_quality"))
        sleep_quality = float(request.form.get("sleep_quality"))
        body_temperature = float(request.form.get("body_temperature"))
        days_since_last_period = int(request.form.get("last_period_day"))

        last_period_date = datetime.today() - timedelta(days=days_since_last_period)
        dayofweek = last_period_date.weekday()
        day = last_period_date.day
        month = last_period_date.month

        input_features = [
            cycle_length, period_length, stress_level, exercise_frequency,
            diet_quality, sleep_quality, body_temperature,
            dayofweek, day, month
        ]

        predicted_days = model.predict([input_features])[0]
        predicted_date = (datetime.today() + timedelta(days=predicted_days)).strftime('%B %d, %Y')

        return render_template("index.html", prediction=f"Your next period is likely on {predicted_date} (in {int(predicted_days)} days).")
    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
