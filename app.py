# app.py

from flask import Flask, request, jsonify
from predict import predict_readmission

app = Flask(__name__)


@app.route('/')
def home():
    """Home endpoint - shows API is running"""
    return """
    <h1>Diabetes Readmission Predictor API</h1>
    
    """


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects JSON with patient data
    Returns prediction result
    """
    try:
        # Get patient data from request
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Make prediction
        result = predict_readmission(patient_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)