import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("T:/Machine-learning_proj/Crop_recommendation.csv")

# Features and label
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
labels = sorted(df['label'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='YlGnBu')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Fertilizer recommendation logic
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
        return ["Fertilizer data not available."]
    ideal_N, ideal_P, ideal_K = ideal_npk[crop]
    tips = []
    if N < ideal_N - 10:
        tips.append("Apply more Urea (Nitrogen)")
    if P < ideal_P - 10:
        tips.append("Apply more DAP (Phosphorus)")
    if K < ideal_K - 10:
        tips.append("Apply more MOP (Potassium)")
    return tips or ["No additional fertilizer needed."]

# Take user input
N = float(input("Enter Nitrogen (N) value: "))
P = float(input("Enter Phosphorus (P) value: "))
K = float(input("Enter Potassium (K) value: "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall (mm): "))

# Create DataFrame from user input
input_values = [N, P, K, temperature, humidity, ph, rainfall]
sample_df = pd.DataFrame([input_values], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

# Scale user input
sample_scaled = scaler.transform(sample_df)

# Predict top 3 crops with confidence
probs = model.predict_proba(sample_scaled)[0]
top3_indices = np.argsort(probs)[::-1][:3]
top3_crops = model.classes_[top3_indices]
top3_confidence = probs[top3_indices]

# Show top 3 crops
print("\nTop 3 Predicted Crops:")
for i in range(3):
    print(f"{i+1}. {top3_crops[i]} ({top3_confidence[i]*100:.2f}%)")

# Fertilizer recommendation for best crop
best_crop = top3_crops[0]
tips = recommend_fertilizer(best_crop, N, P, K)

print(f"\nPredicted Best Crop: {best_crop}")
print("Fertilizer Recommendation:", tips)

