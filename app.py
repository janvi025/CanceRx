from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get single input line
        features_str = request.form['features']

        # Convert string to list of floats
        features = [float(x.strip()) for x in features_str.split(',')]

        # Ensure exactly 30 features
        if len(features) != 31:
            return render_template("index.html", prediction_text="Error: Please enter exactly 30 features separated by commas.")

        final = np.array([features])

        # Make prediction
        prediction = model.predict(final)

        if prediction[0] == 1:
            result = "Cancerous (Malignant)"
        else:
            result = "Not Cancerous (Benign)"

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)