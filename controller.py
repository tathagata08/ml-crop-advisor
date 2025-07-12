from flask import Blueprint, render_template, request
import numpy as np
import pickle

bp = Blueprint('controller', __name__)


model = pickle.load(open("T:\Machine-learning_proj\model\knn_model.pkl", "rb"))
scaler = pickle.load(open("T:\Machine-learning_proj\model\scaler.pkl", "rb"))


ideal_npk = {
    'rice': (90, 40, 40),
    'wheat': (100, 50, 40),
    'maize': (120, 60, 40),
    'cotton': (180, 60, 60),
    'sugarcane': (250, 100, 100),
    'millet': (60, 30, 30),
    'barley': (80, 40, 30),
    'ground nut': (80, 60, 60),
    'soyabean': (100, 60, 60)
}

def recommend_fertilizer(crop, N, P, K):
    crop = crop.lower()
    if crop not in ideal_npk:
        return ["Due to your weather condition your crop need extra care;Please contact with your local Agriculture officer for fertilizer detail"]
    
    ideal_N, ideal_P, ideal_K = ideal_npk[crop]
    messages = []
    if N < ideal_N - 10:
        messages.append("Apply more Urea (Nitrogen)")
    if P < ideal_P - 10:
        messages.append("Apply more DAP (Phosphorus)")
    if K < ideal_K - 10:
        messages.append("Apply more MOP (Potassium)")
    return messages or ["No additional fertilizer needed."]

# Home page
@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None)

# Prediction route
@bp.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from form
        values = [float(request.form[x]) for x in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        input_data = np.array([values])
        scaled_data = scaler.transform(input_data)

        
        proba = model.predict_proba(scaled_data)[0]
        top_index = np.argmax(proba)
        predicted_crop = model.classes_[top_index]
        confidence = round(proba[top_index] * 100, 2)

       
        fertilizer = recommend_fertilizer(predicted_crop, values[0], values[1], values[2])

        return render_template("index.html", prediction=predicted_crop, confidence=confidence, fertilizer=fertilizer)

    except Exception as e:
        return f"Error: {e}"
