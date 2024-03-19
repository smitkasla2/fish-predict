import os
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model_path = "fish_weight_prediction.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the request data
    data = request.get_json(force=True)

    # Prepare the data for prediction
    input_data = pd.DataFrame(data, index=[0])

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    # Use the provided PORT from Heroku's environment, default to 5000 if not provided
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)