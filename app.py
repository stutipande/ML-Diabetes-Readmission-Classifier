from flask import Flask, request, jsonify, render_template
from predict import predict_readmission

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Web UI for manual prediction using HTML form
    """
    if request.method == 'POST':
        try:
            patient_data = dict(request.form)

            # Convert numeric fields
            numeric_fields = [
                "time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses",
                "admission_type_id", "discharge_disposition_id", "admission_source_id"
            ]

            for field in numeric_fields:
                patient_data[field] = int(patient_data[field])

            result = predict_readmission(patient_data)

            return render_template("result.html", result=result)

        except Exception as e:
            return render_template("result.html", error=str(e))

    return render_template("patient_form.html")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for programmatic access
    """
    try:
        patient_data = request.get_json()

        if not patient_data:
            return jsonify({"error": "No patient data provided"}), 400

        result = predict_readmission(patient_data)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
