import gradio as gr
import torch
import numpy as np
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

print("Booting AI on Cloud CPU...")

# Load Binarizer
mlb = joblib.load('genre_binarizer.pkl')

# Load Tokenizer and Model (Notice we don't use .to(device) because the cloud uses CPU)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# We will place our checkpoint files in a folder called 'model_weights' on the cloud
model = DistilBertForSequenceClassification.from_pretrained('./model_weights')
model.eval()

def predict_genre(description):
    # Tokenize
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = torch.sigmoid(outputs.logits).numpy()[0]
    
    # Format output specifically for Gradio's UI
    result_dict = {}
    for idx, genre in enumerate(mlb.classes_):
        result_dict[genre] = float(probs[idx])
        
    return result_dict

# Create the Web Interface
ui = gr.Interface(
    fn=predict_genre,
    inputs=gr.Textbox(lines=5, placeholder="Type a movie description here...", label="Movie Plot"),
    outputs=gr.Label(num_top_classes=5, label="Predicted Genres"),
    title="DistilBERT Movie AI",
    description="I fine-tuned a Transformer neural network to predict movie genres based on raw text. Try it out!"
)

if __name__ == "__main__":
    ui.launch()