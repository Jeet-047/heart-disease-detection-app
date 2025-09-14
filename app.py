from flask import Flask, render_template, request
import joblib
import numpy as np
import os

"""
Flask-based ML app for heart disease prediction with an improved frontend.

Instructions to run:
1. Make sure you have 'flask', 'numpy', and 'scikit-learn' installed:
   pip install Flask numpy scikit-learn
2. Ensure you have the 'model.pkl' file in the same directory.
3. Run this application from your terminal:
   python app.py
4. Open your web browser and go to http://127.0.0.1:5000
"""

app = Flask(__name__)

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
        features = np.array([[age, cp, trestbps, chol, thalach, oldpeak, ca, thal]])

        # Make a prediction
        prediction = model.predict(features)

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
