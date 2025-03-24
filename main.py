from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = FastAPI()

# Load Models and Vectorizer
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
log_model = joblib.load("logistic_model.pkl")
tokenizer = joblib.load("tokenizer.pkl")
lstm_model = load_model("lstm_model.h5")

# Define a Pydantic Model for JSON request
class TextInput(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d{10,}", "<PHONE>", text)  # Mask phone numbers
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<EMAIL>", text)  # Mask emails
    return text.strip()

def predict_sensitive_info(text):
    text_clean = clean_text(text)
    text_tfidf = tfidf_vectorizer.transform([text_clean])
    pred = log_model.predict(text_tfidf)[0]
    
    text_seq = tokenizer.texts_to_sequences([text_clean])
    text_padded = pad_sequences(text_seq, maxlen=100, padding="post")
    lstm_pred = (lstm_model.predict(text_padded) > 0.5).astype("int32")[0][0]
    
    if pred or lstm_pred:
        return {"result": "Your text contains personal details like phone numbers or emails."}
    return {"result": "Success! No sensitive information detected."}

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(data: TextInput):
    return predict_sensitive_info(data.text)
