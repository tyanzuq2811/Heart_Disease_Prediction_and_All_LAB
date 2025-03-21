from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

model = load_model('heart_disease_model.keras')
with open('scaler_params.json', 'r') as f:
    scaler_params = json.load(f)
mean = np.array(scaler_params['mean'])  
std = np.array(scaler_params['std'])   

def get_cp_dummies(cp_value):
    if cp_value == 1: return [1, 0, 0, 0]  
    elif cp_value == 2: return [0, 1, 0, 0]
    elif cp_value == 3: return [0, 0, 1, 0]
    elif cp_value == 4: return [0, 0, 0, 1]
    else: raise ValueError("Invalid cp value")

def get_restecg_dummies(restecg_value):
    if restecg_value == 0: return [0, 0, 1]  
    elif restecg_value == 1: return [0, 1, 0]
    elif restecg_value == 2: return [1, 0, 0]
    else: raise ValueError("Invalid restecg value")

def get_slope_dummies(slope_value):
    if slope_value == 1: return [0, 0, 1]  
    elif slope_value == 2: return [0, 1, 0]
    elif slope_value == 3: return [1, 0, 0]
    else: raise ValueError("Invalid slope value")

def get_thal_dummies(thal_value):
    if thal_value == 1: return [1, 0, 0] 
    elif thal_value == 2: return [0, 1, 0]
    elif thal_value == 3: return [0, 0, 1]
    else: return [0, 0, 0]  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    non_cat_features = [
        float(data['age']), float(data['sex']), float(data['trestbps']),
        float(data['chol']), float(data['fbs']), float(data['thalch']),
        float(data['exang']), float(data['oldpeak']), float(data['ca'])
    ]
    cp_dummies = get_cp_dummies(int(data['cp']))  
    restecg_dummies = get_restecg_dummies(int(data['restecg']))  
    slope_dummies = get_slope_dummies(int(data['slope']))  
    thal_dummies = get_thal_dummies(int(data['thal']))  
    features_list = non_cat_features + cp_dummies + restecg_dummies + slope_dummies + thal_dummies
    features_array = np.array(features_list).reshape(1, -1) 
    print("Features list:", features_list)
    print("Features array shape:", features_array.shape)
    standardized_features = (features_array - mean) / std
    prediction = model.predict(standardized_features)
    pred_class = (prediction > 0.5).astype("int32")[0][0]
    print("Prediction probability:", prediction[0][0])
    print("Predicted class:", pred_class)
    if pred_class == 1:
        result_message = "Có bệnh tim. Hãy liên hệ với bác sĩ để có hướng xử lý kịp thời."
    else:
        result_message = "Không có bệnh tim. Tiếp tục duy trì lối sống lành mạnh."
    return jsonify(result=result_message)

if __name__ == '__main__':
    app.run(debug=True)



    