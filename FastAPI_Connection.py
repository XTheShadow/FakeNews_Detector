from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.data import find
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, or specify your front-end URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)
# Download necessary NLTK resources if not already done
try:
    find('corpora/stopwords.zip')
    find('corpora/wordnet.zip')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load pre-trained models and vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
voting_model = joblib.load('voting_fake_news_model.pkl')


# Text cleaning function
def clean_text(text: str) -> str:
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text


# Define the input data structure
class NewsArticle(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

@app.get("/status")
def status():
    return {"status": "App is running"}


# Prediction endpoint
@app.post("/predict")
async def predict(article: NewsArticle):
    text = article.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned_text = clean_text(text)
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])
    prediction = voting_model.predict(text_vectorized)
    prediction_proba = voting_model.predict_proba(text_vectorized)

    return {
        "label": "Fake" if prediction[0] == 1 else "Real",
        "confidence": round(max(prediction_proba[0]) * 100, 2)
    }


if __name__ == "__main__":
    # Change the "host="127.0.0.1" to "host="0.0.0.0" if this will be running on the cloud
    uvicorn.run(app, host="127.0.0.1", port=8000)

