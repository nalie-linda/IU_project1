
import pandas as pd
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)
# Load the pipeline
with open('model_pipeline_xgb.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)


# Define your feature columns
feature_columns = ['amount', 'fee_success', 'average_fee', 'time_of_day', 'card','3D_secured','last_psp'] 
print(feature_columns)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
     # Create a DataFrame to ensure the correct order of features
    input_data = pd.DataFrame([data], columns=feature_columns)
    # Make prediction
    prediction = model_pipeline.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
