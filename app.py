from flask import Flask, render_template, request
import threading
import urllib.request
import joblib
import numpy as np
import pandas as pd
import os
import yaml
import logging

app = Flask(__name__)

# Global flag to ensure wake-up runs only once
wake_up_ran = False

def wake_up_other_apps():
    model_info = {}
    with open("model_info.yaml", "r") as f:
        model_info = yaml.safe_load(f)
    
    urls = model_info.get("URLS", {})
    for app_name, url in urls.items():
        try:
            logging.info(f"Waking up {app_name} at {url}")
            urllib.request.urlopen(url, timeout=120)
            logging.info(f"{app_name} is awake!")
        except Exception as e:
            logging.error(f"Failed to wake up {app_name}: {e}")

@app.before_request
def trigger_wake_up():
    global wake_up_ran
    if not wake_up_ran:
        wake_up_ran = True
        # run wake up in a background thread so the request is not slowed down
        threading.Thread(target=wake_up_other_apps, daemon=True).start()

# Try to load the model. If the file doesn't exist, provide a clear error.
try:
    with open('heart_disease_rf.pkl', 'rb') as file:
        model = joblib.load(file)
        print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Error: model.pkl not found. Please place your trained model file in the project directory.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the model: {e}")

@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200

@app.route('/')
def home():
    """
    Renders the beautiful home page.
    """
    return render_template('index.html')

@app.route('/form')
def show_form():
    """
    Renders the prediction form page.
    """
    return render_template('prediction_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission and makes a prediction.
    """
    if not model:
        return "Model not found. Please check your setup.", 500

    # Extracting and converting form data
    try:
        patient_name = request.form['patient_name']
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        thalach = int(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create a numpy array with the input features
        feature_names = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
        features = np.array([[age, cp, trestbps, chol, thalach, oldpeak, ca, thal]])

        # Create a dataframe
        input_df = pd.DataFrame(features, columns=feature_names)

        # Make a prediction
        prediction = model.predict(input_df)

        result_text = "Presence of Heart Disease" if prediction[0] == 0 else "No Heart Disease"

        # Pass all collected data and the prediction to the report template
        patient_data = {
            'name': patient_name,
            'age': age,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'ca': ca,
            'thal': thal,
            'prediction': result_text
        }

        return render_template('report.html', patient_data=patient_data)

    except (ValueError, KeyError) as e:
        return f"Invalid input. Please check the values. Error: {e}", 400
    except Exception as e:
        return f"An unexpected error occurred: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
