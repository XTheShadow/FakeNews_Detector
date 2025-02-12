import nltk
import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from nltk.corpus import stopwords
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from nltk.stem import WordNetLemmatizer
from nltk.data import find
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK resources if not already done
try:
    find('corpora/stopwords.zip')
    find('corpora/wordnet.zip')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')


# Modify the data loading to use only a subset of rows for each dataset
# Define the number of rows to load from each dataset
subset_size = 1500  # You can adjust this value based on memory and processing limits

# Load the LIAR dataset files
liar_train_path = "datasets/LIAR_Dataset/train.tsv"
liar_test_path = "datasets/LIAR_Dataset/test.tsv"
liar_valid_path = "datasets/LIAR_Dataset/valid.tsv"

# Only load a subset of each file
df_train_liar = pd.read_csv(liar_train_path, sep='\t', header=None, nrows=subset_size)
df_test_liar = pd.read_csv(liar_test_path, sep='\t', header=None, nrows=subset_size)
df_valid_liar = pd.read_csv(liar_valid_path, sep='\t', header=None, nrows=subset_size)

# Combine train, test, and validation sets
df_liar = pd.concat([df_train_liar, df_test_liar, df_valid_liar], ignore_index=True)

# Assign column names to the LIAR dataset
df_liar.columns = [
    'ID', 'label', 'statement', 'subject', 'speaker', 'job', 'state_info',
    'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
]

# Handle missing values in LIAR dataset
df_liar.fillna({'statement': '', 'label': 1}, inplace=True)

# Map labels to binary (0 for real, 1 for fake)
def map_liar_labels(label):
    return 0 if label in ['true', 'mostly-true'] else 1

df_liar['label'] = df_liar['label'].apply(map_liar_labels)
df_liar['text'] = df_liar['statement']

# Reduce the size of the LIAR dataset
df_liar = df_liar.sample(n=min(subset_size, len(df_liar)), random_state=42)

# Select relevant columns for the LIAR dataset
df_liar = df_liar[['text', 'label']]

# Load the ISOT datasets (True.csv and Fake.csv)
df_true = pd.read_csv('datasets/ISOT_Dataset/True.csv', nrows=subset_size)
df_fake = pd.read_csv('datasets/ISOT_Dataset/Fake.csv', nrows=subset_size)

# Add label column: 0 for real news, 1 for fake news
df_true['label'] = 0
df_fake['label'] = 1

# Combine ISOT datasets
df_isot = pd.concat([df_true, df_fake], ignore_index=True)
df_isot = df_isot[['text', 'label']]

# Load the Fake_Real.csv dataset
df_fake_real = pd.read_csv('datasets/Fake-Real_Dataset/Fake_Real.csv', nrows=subset_size)

# Map 'target' column to binary labels
df_fake_real['label'] = df_fake_real['target'].map({'True': 0, 'Fake': 1})

# Drop unnecessary columns in Fake_Real and keep only text and label
df_fake_real = df_fake_real[['text', 'label']]

# Load the fake_news.csv dataset
df_fake_news = pd.read_csv('fake_news.csv', nrows=subset_size)

# Map 'label' column in fake_news.csv to binary (1 for FAKE, 0 for REAL)
df_fake_news['label'] = df_fake_news['label'].map({'FAKE': 1, 'REAL': 0})

# Select relevant columns from fake_news.csv
df_fake_news = df_fake_news[['text', 'label']]

# Load the new datasets (train1.csv, test1.csv, evaluation1.csv)
df_train_1 = pd.read_csv('datasets/Dataset/train1.csv', sep=';', header=0, nrows=subset_size)
df_test_1 = pd.read_csv('datasets/Dataset/test1.csv', sep=';', header=0, nrows=subset_size)
df_eval_1 = pd.read_csv('datasets/Dataset/evaluation.csv', sep=';', header=0, nrows=subset_size)

# Combine the new train, test, and evaluation datasets
df_1 = pd.concat([df_train_1, df_test_1, df_eval_1], ignore_index=True)

# Reverse the label mappings for train1, test1, and evaluation datasets
df_1['label'] = df_1['label'].map({1: 0, 0: 1})

# Handle missing values in the new dataset
df_1.fillna({'text': '', 'label': 1}, inplace=True)

# Select relevant columns for the first new dataset
df_1 = df_1[['text', 'label']]

# Load the second new dataset (true1.csv and fake1.csv)
df_true_2 = pd.read_csv('datasets/Dataset1/true1.csv', nrows=subset_size)
df_fake_2 = pd.read_csv('datasets/Dataset/fake1.csv', nrows=subset_size)

# Add label column: 0 for real news, 1 for fake news
df_true_2['label'] = 0
df_fake_2['label'] = 1

# Combine the second new dataset
df_2 = pd.concat([df_true_2, df_fake_2], ignore_index=True)
df_2 = df_2[['text', 'label']]

# Combine all datasets into one
df_combined = pd.concat([df_isot, df_liar, df_fake_real, df_fake_news, df_1, df_2], ignore_index=True)

# Check for any remaining missing values in the combined dataset
print("Missing values in combined dataset:", df_combined.isnull().sum())

# Drop rows with missing 'text' or 'label'
df_combined.dropna(subset=['text', 'label'], inplace=True)

# Remove duplicate rows
df_combined.drop_duplicates(subset=['text'], inplace=True)

# Data cleaning function for text
def clean_text(text):
    # Remove non-alphanumeric characters and digits
    text = re.sub(r'\W', ' ', text)
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])  # Lemmatization and stopword removal
    return text

# Apply text cleaning function
df_combined['cleaned_text'] = df_combined['text'].apply(clean_text)

# Define features and labels
X = df_combined['cleaned_text']
y = df_combined['label']

# TF-IDF Vectorizer for text
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
X_text_tfidf = tfidf_vectorizer.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_text_tfidf, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Function to train and evaluate different models
def train_and_evaluate_model(model, model_name):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

    # Check if the model supports predict_proba before calculating ROC AUC
    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f'ROC AUC Score: {roc_auc}')
    else:
        print(f'ROC AUC Score: Not available for {model_name}')

    return model


# Train and evaluate Logistic Regression with GridSearchCV
logistic_model = LogisticRegression(max_iter=1000)
logistic_param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
logistic_grid_search = GridSearchCV(logistic_model, logistic_param_grid, cv=5)
logistic_model = train_and_evaluate_model(logistic_grid_search, "Logistic Regression")

# Train and evaluate Ridge Classifier with GridSearchCV
ridge_model = RidgeClassifier()
ridge_param_grid = {'alpha': [0.1, 1, 10]}
ridge_grid_search = GridSearchCV(ridge_model, ridge_param_grid, cv=5)
ridge_model = train_and_evaluate_model(ridge_grid_search, "Ridge Classifier")

# Train and evaluate Random Forest Classifier with GridSearchCV
random_forest_model = RandomForestClassifier(random_state=42)
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5]}
rf_grid_search = GridSearchCV(random_forest_model, rf_param_grid, cv=5)
random_forest_model = train_and_evaluate_model(rf_grid_search, "Random Forest Classifier")


# Voting Classifier (Ensemble of Logistic, Ridge, and Random Forest)
voting_model = VotingClassifier(estimators=[
    ('logistic', logistic_grid_search),
    ('rf', rf_grid_search)
], voting='soft')  # Soft voting relies on predict_proba


# Train and evaluate the voting classifier
voting_model = train_and_evaluate_model(voting_model, "Voting Classifier")

# Save models and vectorizer
joblib.dump(logistic_model, 'logistic_fake_news_model.pkl')
joblib.dump(ridge_model, 'ridge_fake_news_model.pkl')
joblib.dump(random_forest_model, 'random_forest_fake_news_model.pkl')
joblib.dump(voting_model, 'voting_fake_news_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Predict news function
def predict_news(text):
    cleaned_text = clean_text(text)
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])
    prediction = voting_model.predict(text_vectorized)
    prediction_proba = voting_model.predict_proba(text_vectorized)
    proba_percentage = round(max(prediction_proba[0]) * 100, 2)
    label = "Fake" if prediction[0] == 1 else "Real"
    result = f'Prediction: {label}\nConfidence: {proba_percentage}%'
    return result


# Confusion Matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ROC Curve
def plot_roc_curve(y_test, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')  # Random guess line
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Feature Importance for Random Forest
def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.title("Top 10 Feature Importances")
        plt.show()
    else:
        print(f"{model.__class__.__name__} does not support feature importance.")

# Example Usage During Model Evaluation
y_pred = logistic_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred, "Logistic Regression")

if hasattr(logistic_model, "predict_proba"):
    y_score = logistic_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_score, "Logistic Regression")

# For Random Forest Feature Importance
plot_feature_importance(random_forest_model, tfidf_vectorizer.get_feature_names_out())

# Predict with the Voting Classifier
y_pred_voting = voting_model.predict(X_test)

# Plot Confusion Matrix for Voting Classifier
plot_confusion_matrix(y_test, y_pred_voting, "Voting Classifier")

# If the Voting Classifier supports predict_proba, plot ROC Curve as well
if hasattr(voting_model, "predict_proba"):
    y_score_voting = voting_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_score_voting, "Voting Classifier")
