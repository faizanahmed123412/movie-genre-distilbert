from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import glob
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ==========================================
# 1. DEFINE API DATA STRUCTURES
# ==========================================
class MovieRequest(BaseModel):
    description: str
    threshold: float = 0.5  # Defaults to 0.5 if the user doesn't provide one

class GenrePrediction(BaseModel):
    genre: str
    confidence_percent: float

class MovieResponse(BaseModel):
    input_text: str
    predictions: list[GenrePrediction]

# ==========================================
# 2. INITIALIZE AND LOAD THE AI (RUNS ONCE)
# ==========================================
app = FastAPI(title="DistilBERT Movie Genre Classifier API")

print("Booting AI Engine...")
# Load the saved genre mapping
mlb = joblib.load('genre_binarizer.pkl')

# Locate and load the latest weights
checkpoints = glob.glob('./results/checkpoint-*')
if not checkpoints:
    raise RuntimeError("No model checkpoints found in ./results/")
latest_checkpoint = max(checkpoints, key=os.path.getctime)

# Load Tokenizer and Model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(latest_checkpoint)

# Move to GPU if available, lock for evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"Engine Online. Running on: {device.type.upper()}")

# ==========================================
# 3. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/predict", response_model=MovieResponse)
async def predict_genre(request: MovieRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty.")
    
    # Tokenize and move to device
    inputs = tokenizer(
        request.description,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Run Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Process Probabilities
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Extract passing genres
    results = []
    for idx, genre in enumerate(mlb.classes_):
        confidence = probs[idx]
        if confidence >= request.threshold:
            results.append(
                GenrePrediction(
                    genre=genre, 
                    confidence_percent=round(float(confidence * 100), 2)
                )
            )
            
    # Sort results by confidence (highest first)
    results.sort(key=lambda x: x.confidence_percent, reverse=True)
    
    return MovieResponse(
        input_text=request.description,
        predictions=results
    )