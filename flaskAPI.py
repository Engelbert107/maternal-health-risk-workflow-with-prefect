import joblib
import pandas as pd
from flasgger import Swagger
from flask import Flask, request


app = Flask(__name__)
Swagger(app)

knn_loaded = joblib.load("models/final_knn_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
power_transformer = joblib.load("models/power_transformer.pkl")

@app.route('/predict', methods=["GET"])
def predict_class():
    """ 
    Predict whether a woman is at low or elevated risk of sickness during pregnancy.
    ---
    parameters:
      - name: Age   
        in: query
        type: integer
        required: true
        description: "Age of the woman in years."
      - name: SystolicBP    
        in: query
        type: integer
        required: true
        description: "Upper value of Blood Pressure (in mmHg)."
      - name: DiastolicBP
        in: query
        type: integer
        required: true
        description: "Lower value of Blood Pressure (in mmHg)."
      - name: BS
        in: query
        type: number
        format: float  
        required: true
        description: "Blood glucose levels (in mmol/L)."
      - name: BodyTemp
        in: query
        type: number
        format: float  
        required: true
        description: "Body temperature (in Fahrenheit)."
      - name: HeartRate
        in: query
        type: integer
        required: true
        description: "Normal resting heart rate (in beats per minute)."
    responses:
      500:
        description: Single Prediction
    """
    try:
        Age = request.args.get("Age", default=35, type=int)
        SystolicBP = request.args.get("SystolicBP", default=120, type=int)
        DiastolicBP = request.args.get("DiastolicBP", default=60, type=int)
        BS = request.args.get("BS", default=6.1, type=float)
        BodyTemp = request.args.get("BodyTemp", default=98.0, type=float)
        HeartRate = request.args.get("HeartRate", default=76, type=int)

        if None in [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]:
            return "Error: Missing one or more required parameters", 400
        
        input_data = pd.DataFrame([[Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]], columns=["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"])
        
        transform_columns = ["HeartRate", "BodyTemp", "BS"]
        input_data[transform_columns] = power_transformer.transform(input_data[transform_columns])

        input_data_scaled = scaler.transform(input_data)

        prediction = knn_loaded.predict(input_data_scaled)
        
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        return f"The risk level of this woman during pregnancy is: {prediction_label}"

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/predict_file', methods=["POST"])
def prediction_test_file():
    """ 
    Predict whether women are at low or elevated risk of sickness during pregnancy.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: |
          The CSV file should contain the following columns:
          - **Age**: Age of the woman during pregnancy (integer)
          - **SystolicBP**: Upper value of Blood Pressure (integer)
          - **DiastolicBP**: Lower value of Blood Pressure (integer)
          - **BS**: Blood glucose levels (float)
          - **BodyTemp**: Body Temperature (float)
          - **HeartRate**: Normal resting heart rate (integer)
          The file should not contain missing values for these columns.
    responses:
      500:
        description: Error handling for the file and prediction process.
      200:
        description: A list of risk level predictions ('low risk' or 'elevated risk') for each record in the file.
    """
    if 'file' not in request.files:
        return "Error: No file part", 400

    file = request.files['file']
    
    if file.filename == '':
        return "Error: No selected file", 400
    
    if not file.filename.endswith('.csv'):
        return "Error: The file must be a CSV", 400

    try:
        test_data = pd.read_csv(file)

        required_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        if not all(col in test_data.columns for col in required_columns):
            return f"Error: The input CSV is missing required columns: {', '.join(required_columns)}", 400
          
        transform_columns = ["HeartRate", "BodyTemp", "BS"]
        test_data[transform_columns] = power_transformer.transform(test_data[transform_columns])

        test_data_scaled = scaler.transform(test_data)

        prediction = knn_loaded.predict(test_data_scaled)
        prediction_labels = label_encoder.inverse_transform(prediction)

        return str(list(prediction_labels))

    except Exception as e:
        return f"Error: {str(e)}", 500


""" 
Open the browser on: http://localhost:5000/apidocs
"""


if __name__=="__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
