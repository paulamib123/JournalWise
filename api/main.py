GEMINI_API_KEY="" # enter your gemini api key here
    
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from google.genai import types, Client

# to run use this command - uvicorn main:app --reload

model = joblib.load("../models/logistic_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

app = FastAPI()
client = Client(api_key=GEMINI_API_KEY)

class_to_labels = {
    "0": "catastrophizing",
    "1": "emotionally reasoning",
    "2": "mind reading",
    "3": "overgeneralizing",
    "4": "personalizing"
}

class JournalEntry(BaseModel):
    text: str

@app.get("/")
def show_message():
    return {"message": "Welcome to the Journal Emotion Analysis API"}

def reframe_thought(distortion_label: str, entry_text: str) -> str:
    prompt = (
        f"You are a compassionate cognitive distortion reframing assistant.\n"
        f"Original journal entry:\n\n{entry_text}\n\n"
        f"Cognitive distortion type detected: **{distortion_label}**.\n"
        "Please provide a constructive reframing of this thought."
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
    )
    return response.candidates[0].content.parts[0].text

@app.post("/predict")
def predict_emotion(entry: JournalEntry):
    vect = vectorizer.transform([entry.text])
    probs = model.predict_proba(vect)
    if probs is None:
        return {"error": "Could not predict distortion type"}
    
    probability = float(np.max(probs))
    class_label = str(int(np.argmax(probs)))
    label = class_to_labels[class_label]

    if probability < 0.5:
        return {"prediction": "**No distortion detected**", "probability": probability}
    
    reframed = reframe_thought(label, entry.text)
    return {
        "prediction": label,
        "probability": probability,
        "reframed_thought": reframed
    }
