import pickle
print("Pickle is available.")
# predict_crop.py



import numpy as np

# Load saved model and scaler
with open("model/knn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Get input from user
N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temp = float(input("Enter Temperature (°C): "))
humid = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH: "))
rain = float(input("Enter Rainfall (mm): "))

# Create and scale sample
sample = np.array([[N, P, K, temp, humid, ph, rain]])
sample_scaled = scaler.transform(sample)

# Predict crop and confidence
predicted_crop = model.predict(sample_scaled)[0]
proba = model.predict_proba(sample_scaled)[0]  # returns list of probabilities
confidence = max(proba) * 100  # convert to percentage

print(f"\n🌾 Recommended Crop: {predicted_crop}")
print(f"🔍 Confidence: {confidence:.2f}%")

# Optional: Show top 3 predictions
classes = model.classes_
top_indices = proba.argsort()[-3:][::-1]  # indices of top 3 crops
print("\n📊 Top 3 Crop Suggestions:")
for i in top_indices:
    print(f"{classes[i]} — {proba[i]*100:.2f}%")
