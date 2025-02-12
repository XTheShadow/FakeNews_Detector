# 🤖 FakeNews_Detector

## 📖 Description  
FakeNews_Detector is a **Python-based** news classification system that uses **machine learning models** to identify fake news articles. It combines multiple datasets and employs an ensemble voting classifier for improved accuracy. The system features a **FastAPI backend** for real-time predictions and includes comprehensive model evaluation visualizations.

---

## 📌 Features  
✔️ **Multi-dataset training** (LIAR, ISOT, Fake-Real, and custom datasets)  
✔️ **Advanced text preprocessing** with lemmatization and TF-IDF vectorization  
✔️ **Class imbalance handling** using SMOTE  
✔️ **Hyperparameter tuning** with GridSearchCV  
✔️ **Ensemble learning** with Voting Classifier  
✔️ **Model evaluation metrics** (Confusion Matrix, ROC Curve, Feature Importance)  
✔️ **FastAPI backend** with CORS support  
✔️ **Probability-based predictions** with confidence scores  

---


## 🛠️ Installation  

### ✅ Prerequisites  
- 🐍 **Python 3.8** or later
- 📚 Essential libraries:
  ```bash
  FastAPI, Uvicorn, Scikit-learn, Pandas, Numpy, Joblib, NLTK, imbalanced-learn, matplotlib, seaborn

  
## 🚀 Setup & Installation

### 1️⃣ Clone the repository:
```bash
git clone https://github.com/XTheShadow/FakeNews_Detector.git
cd FakeNews_Detector
```


### 2️⃣ Set up the virtual environment:
```bash
python -m venv .venv
```


#### Windows:
```bash
.\.venv\Scripts\activate
```

#### macOS/Linux:
```bash
source .venv/bin/activate
```


### 3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```


### 4️⃣ Train the model:
```bash
python model.py
```
This will:

- Preprocess all datasets
- Train multiple models with hyperparameter tuning
- Generate evaluation plots
- Save trained models (`.pkl` files) and vectorizer


### 5️⃣ Launch FastAPI server:
```bash
uvicorn FastAPI_Connection:app --reload
```



## 🎮 API Usage
The API provides two endpoints:

### 🔍 Prediction Endpoint
**URL:** POST /predict

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "label": "Fake/Real",
  "confidence": 95.23
}
```


### 🌐 Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Scientists confirm COVID-19 was developed in a lab as biological weapon"}'
```



## 🧠 Model Architecture

### 🔧 Data Pipeline
  
* **Text Preprocessing:**
  - Special character removal
  - Lemmatization with NLTK
  - Stopword removal
  - TF-IDF vectorization (5000 features)

* **Class Balancing:**
  - SMOTE oversampling for imbalanced classes


### 🤖 Machine Learning Models

- Logistic Regression (with GridSearchCV tuning)
- Ridge Classifier
- Random Forest
- Ensemble Voting Classifier


### 📊 Evaluation Metrics

- Accuracy scores
- ROC-AUC curves
- Confusion matrices
- Feature importance visualization



## 📂 Project Structure

```
├── datasets/
│   ├── LIAR_Dataset/       # LIAR dataset files
│   ├── ISOT_Dataset/       # True/Fake news datasets
│   └── ...                 # Other datasets
├── model.py                # Main training script
├── FastAPI_Connection.py   # API implementation
├── requirements.txt        # Dependencies
└── README.md               # This document
```



## 📜 License
This project is licensed under the MIT License.
