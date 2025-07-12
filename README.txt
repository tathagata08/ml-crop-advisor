

# 🌾 Crop Recommendation using Machine Learning

This project uses machine learning techniques to recommend the most suitable crop for cultivation based on soil and environmental conditions. It aims to help farmers and agri-consultants make informed decisions to improve yield and resource efficiency.

---

## 📊 Dataset

The model is trained on a dataset with the following features:

- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **Temperature**: In degrees Celsius
- **Humidity**: Percentage
- **pH**: Acidity/alkalinity level of the soil
- **Rainfall**: In mm
- **Label**: Recommended crop (target variable)

---

## 🧠 Machine Learning Approach

- **Type**: Supervised Classification
- **Algorithms Explored**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Evaluation Metrics**: Accuracy, Precision, Recall, Confusion Matrix

---

## 📈 Results

The best-performing model achieved over 95% accuracy on the test dataset. Random Forest and SVM provided the most reliable recommendations.

---

## 🔧 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crop-recommender-ml.git
   cd crop-recommender-ml


#### How to run the PROJECT::
Run the train_model.py it will create 2 files in model folder 
This folder is used to store trained data , it will create them


## 🏋️‍♂️ Training the Model (`train_model.py`)

This script trains a K-Nearest Neighbors (KNN) classification model on the crop recommendation dataset using soil and climate features such as N, P, K, temperature, humidity, pH, and rainfall.

### 🔧 Functionality

- Loads the dataset from `Crop_recommendation.csv`
- Scales the input features using `StandardScaler`
- Splits the dataset into training and testing sets
- Trains a KNN model with `k=3`
- Saves the trained model (`knn_model.pkl`) and the scaler (`scaler.pkl`) inside a `model/` directory

### 🧪 How to Run

```bash
python train_model.py


THEN ==>>

** HOW TO RUN THE WEBAPP **


1.To see the full flask app run the app.py --> Then go to browser and log into the local host i.e http://127.0.0.1:5000/
It will render a flask app where user can put N P K RAINFALL HUMIDITY and other required values to get the desired output



            @TATHAGATA BANERJEE (tathagatabanerjee2005@gmail.com)