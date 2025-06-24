"""
Backend server for DASS-42 questionnaire with KNN prediction models
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the KNN models
try:
    depression_model = joblib.load('models/dp_k_model.pkl')
    anxiety_model = joblib.load('models/ax_k_model.pkl')
    stress_model = joblib.load('models/st_k_model.pkl')
    
    # Get feature names from each model
    depression_features = depression_model.feature_names_in_
    anxiety_features = anxiety_model.feature_names_in_
    stress_features = stress_model.feature_names_in_
    
    print("Depression model features:", depression_features)
    print("Anxiety model features:", anxiety_features)
    print("Stress model features:", stress_features)
except Exception as e:
    print(f"Warning: Error loading models: {str(e)}")
    depression_model = None
    anxiety_model = None
    stress_model = None
    depression_features = None
    anxiety_features = None
    stress_features = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/test')
def questionnaire():
    """Render the questionnaire page"""
    return render_template('questionnaire.html')

@app.route('/results')
def results():
    """Render the results page"""
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions based on DASS-42 questionnaire responses
    
    Expected input: JSON with answers array containing 42 values (0-3)
    Returns: Prediction results for Depression, Anxiety, and Stress levels
    """
    try:
        data = request.json
        answers = data.get('answers')
        
        if not answers or len(answers) != 42:
            return jsonify({
                'error': 'Invalid input: Exactly 42 answers required'
            }), 400
            
        if any(features is None for features in [depression_features, anxiety_features, stress_features]):
            return jsonify({
                'error': 'Model feature names not loaded'
            }), 500
            
        try:
            # Create a mapping of question numbers to answers, converting strings to integers
            # and handling None values by defaulting to 0
            answer_dict = {}
            for i, ans in enumerate(answers):
                try:
                    # Convert string to integer if possible, otherwise use 0
                    value = int(ans) if ans is not None else 0
                    answer_dict[f'Q{i+1}A'] = value
                except (ValueError, TypeError):
                    answer_dict[f'Q{i+1}A'] = 0
            
            print("Answer dictionary:", answer_dict)
            
            # Create DataFrames with only the required features for each model
            # Ensure all required features are present and in the correct order
            depression_X = pd.DataFrame([{k: answer_dict.get(k, 0) for k in depression_features}], 
                                     columns=depression_features)
            anxiety_X = pd.DataFrame([{k: answer_dict.get(k, 0) for k in anxiety_features}], 
                                   columns=anxiety_features)
            stress_X = pd.DataFrame([{k: answer_dict.get(k, 0) for k in stress_features}], 
                                  columns=stress_features)
            
            print("Depression features:", depression_features)
            print("Depression DataFrame:", depression_X)
            print("Anxiety features:", anxiety_features)
            print("Anxiety DataFrame:", anxiety_X)
            print("Stress features:", stress_features)
            print("Stress DataFrame:", stress_X)
            
            # Verify no NaN values
            if depression_X.isna().any().any():
                print("NaN values in depression DataFrame:", depression_X.isna().sum())
                raise ValueError("NaN values detected in depression data")
            if anxiety_X.isna().any().any():
                print("NaN values in anxiety DataFrame:", anxiety_X.isna().sum())
                raise ValueError("NaN values detected in anxiety data")
            if stress_X.isna().any().any():
                print("NaN values in stress DataFrame:", stress_X.isna().sum())
                raise ValueError("NaN values detected in stress data")
            
            # Make predictions using each model
            depression_pred = depression_model.predict(depression_X)[0]
            anxiety_pred = anxiety_model.predict(anxiety_X)[0]
            stress_pred = stress_model.predict(stress_X)[0]
            
        except Exception as model_error:
            print(f"Model prediction error: {str(model_error)}")
            print(f"Depression features: {depression_features}")
            print(f"Anxiety features: {anxiety_features}")
            print(f"Stress features: {stress_features}")
            return jsonify({
                'error': 'Error during model prediction'
            }), 500
        
        # Calculate scores for each category
        depression_score = sum(int(answers[i-1]) if answers[i-1] is not None else 0 
                             for i in range(3, 42, 3))  # Questions 3, 6, 9, ..., 42
        anxiety_score = sum(int(answers[i-1]) if answers[i-1] is not None else 0 
                           for i in range(2, 42, 3))    # Questions 2, 5, 8, ..., 41
        stress_score = sum(int(answers[i-1]) if answers[i-1] is not None else 0 
                          for i in range(1, 42, 3))     # Questions 1, 4, 7, ..., 40
        
        def depression_severity(score):
            """Return severity label for depression score (DASS-42)"""
            if score <= 9:
                return "Normal"
            elif score <= 13:
                return "Mild"
            elif score <= 20:
                return "Moderate"
            elif score <= 27:
                return "Severe"
            else:
                return "Extremely Severe"

        def anxiety_severity(score):
            """Return severity label for anxiety score (DASS-42)"""
            if score <= 7:
                return "Normal"
            elif score <= 9:
                return "Mild"
            elif score <= 14:
                return "Moderate"
            elif score <= 19:
                return "Severe"
            else:
                return "Extremely Severe"

        def stress_severity(score):
            """Return severity label for stress score (DASS-42)"""
            if score <= 14:
                return "Normal"
            elif score <= 18:
                return "Mild"
            elif score <= 25:
                return "Moderate"
            elif score <= 33:
                return "Severe"
            else:
                return "Extremely Severe"

        result = {
            'depression': depression_severity(depression_score),
            'anxiety': anxiety_severity(anxiety_score),
            'stress': stress_severity(stress_score),
            'scores': {
                'depression': depression_score,
                'anxiety': anxiety_score,
                'stress': stress_score
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 