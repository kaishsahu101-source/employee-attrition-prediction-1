"""
Employee Attrition Prediction - Web Application using Flask
This script creates a web interface for predicting employee attrition.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    model_data = joblib.load('models/attrition_model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to create the model.")
    model = None

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create input dataframe
        input_data = {
            'Age': int(data.get('Age', 30)),
            'BusinessTravel': data.get('BusinessTravel', 'Travel_Rarely'),
            'DailyRate': int(data.get('DailyRate', 800)),
            'Department': data.get('Department', 'Sales'),
            'DistanceFromHome': int(data.get('DistanceFromHome', 5)),
            'Education': int(data.get('Education', 3)),
            'EducationField': data.get('EducationField', 'Life Sciences'),
            'EnvironmentSatisfaction': int(data.get('EnvironmentSatisfaction', 3)),
            'Gender': data.get('Gender', 'Male'),
            'HourlyRate': int(data.get('HourlyRate', 50)),
            'JobInvolvement': int(data.get('JobInvolvement', 3)),
            'JobLevel': int(data.get('JobLevel', 2)),
            'JobRole': data.get('JobRole', 'Sales Executive'),
            'JobSatisfaction': int(data.get('JobSatisfaction', 3)),
            'MaritalStatus': data.get('MaritalStatus', 'Single'),
            'MonthlyIncome': int(data.get('MonthlyIncome', 5000)),
            'MonthlyRate': int(data.get('MonthlyRate', 10000)),
            'NumCompaniesWorked': int(data.get('NumCompaniesWorked', 2)),
            'OverTime': data.get('OverTime', 'No'),
            'PercentSalaryHike': int(data.get('PercentSalaryHike', 15)),
            'PerformanceRating': int(data.get('PerformanceRating', 3)),
            'RelationshipSatisfaction': int(data.get('RelationshipSatisfaction', 3)),
            'StockOptionLevel': int(data.get('StockOptionLevel', 1)),
            'TotalWorkingYears': int(data.get('TotalWorkingYears', 5)),
            'TrainingTimesLastYear': int(data.get('TrainingTimesLastYear', 3)),
            'WorkLifeBalance': int(data.get('WorkLifeBalance', 3)),
            'YearsAtCompany': int(data.get('YearsAtCompany', 3)),
            'YearsInCurrentRole': int(data.get('YearsInCurrentRole', 2)),
            'YearsSinceLastPromotion': int(data.get('YearsSinceLastPromotion', 2)),
            'YearsWithCurrManager': int(data.get('YearsWithCurrManager', 2))
        }
        
        # Create DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Encode categorical variables using saved label encoders
        for col, encoder in label_encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = encoder.transform(df_input[col])
                except:
                    # If unseen label, use the most frequent class
                    df_input[col] = 0
        
        # Scale features
        X_input = scaler.transform(df_input)
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]
        
        # Prepare result
        result = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(probability * 100, 2),
            'risk_level': 'High' if probability > 0.7 else ('Medium' if probability > 0.4 else 'Low')
        }
        
        return render_template('index.html', 
                             prediction=result['prediction'],
                             probability=result['probability'],
                             risk_level=result['risk_level'],
                             form_data=data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (JSON input/output)"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Similar processing as /predict endpoint
        input_data = {
            'Age': int(data.get('Age', 30)),
            'BusinessTravel': data.get('BusinessTravel', 'Travel_Rarely'),
            'DailyRate': int(data.get('DailyRate', 800)),
            'Department': data.get('Department', 'Sales'),
            'DistanceFromHome': int(data.get('DistanceFromHome', 5)),
            'Education': int(data.get('Education', 3)),
            'EducationField': data.get('EducationField', 'Life Sciences'),
            'EnvironmentSatisfaction': int(data.get('EnvironmentSatisfaction', 3)),
            'Gender': data.get('Gender', 'Male'),
            'HourlyRate': int(data.get('HourlyRate', 50)),
            'JobInvolvement': int(data.get('JobInvolvement', 3)),
            'JobLevel': int(data.get('JobLevel', 2)),
            'JobRole': data.get('JobRole', 'Sales Executive'),
            'JobSatisfaction': int(data.get('JobSatisfaction', 3)),
            'MaritalStatus': data.get('MaritalStatus', 'Single'),
            'MonthlyIncome': int(data.get('MonthlyIncome', 5000)),
            'MonthlyRate': int(data.get('MonthlyRate', 10000)),
            'NumCompaniesWorked': int(data.get('NumCompaniesWorked', 2)),
            'OverTime': data.get('OverTime', 'No'),
            'PercentSalaryHike': int(data.get('PercentSalaryHike', 15)),
            'PerformanceRating': int(data.get('PerformanceRating', 3)),
            'RelationshipSatisfaction': int(data.get('RelationshipSatisfaction', 3)),
            'StockOptionLevel': int(data.get('StockOptionLevel', 1)),
            'TotalWorkingYears': int(data.get('TotalWorkingYears', 5)),
            'TrainingTimesLastYear': int(data.get('TrainingTimesLastYear', 3)),
            'WorkLifeBalance': int(data.get('WorkLifeBalance', 3)),
            'YearsAtCompany': int(data.get('YearsAtCompany', 3)),
            'YearsInCurrentRole': int(data.get('YearsInCurrentRole', 2)),
            'YearsSinceLastPromotion': int(data.get('YearsSinceLastPromotion', 2)),
            'YearsWithCurrManager': int(data.get('YearsWithCurrManager', 2))
        }
        
        df_input = pd.DataFrame([input_data])
        
        for col, encoder in label_encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = encoder.transform(df_input[col])
                except:
                    df_input[col] = 0
        
        X_input = scaler.transform(df_input)
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]
        
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(probability * 100, 2),
            'risk_level': 'High' if probability > 0.7 else ('Medium' if probability > 0.4 else 'Low')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
