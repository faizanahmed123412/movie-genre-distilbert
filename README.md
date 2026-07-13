# 🎬 DistilBERT Multi-Label Movie Genre Classifier

An end-to-end Deep Learning pipeline that fine-tunes a **DistilBERT Transformer neural network** using PyTorch to predict multi-label movie genres from raw plot summaries.

🚀 **Live Production Demo:** [View Live App on Hugging Face Spaces](https://felixastr1245-movie-genre-ai.hf.space)

---

## 🛠️ Tech Stack & Architecture

- **Model Core:** Pre-trained `distilbert-base-uncased` (Hugging Face Transformers) fine-tuned locally using PyTorch.
- **Hardware Optimization:** Mixed-precision (FP16) training executed on a local NVIDIA RTX 4050 GPU using CUDA.
- **Backend API:** FastAPI & Uvicorn providing a highly concurrent production REST endpoint.
- **Cloud Deployment:** Hosted on Hugging Face Spaces with a dynamic web UI powered by Gradio.

---

## 📁 Repository Structure

- `program1.py`: The production FastAPI web server implementation.
- `app.py`: The Gradio web interface client built for cloud deployment.
- `genre_binarizer.pkl`: Serialized `MultiLabelBinarizer` for decoding text class indices.
- `requirements.txt`: Application dependencies.

---

## 🚀 Local Deployment (FastAPI Web Server)

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/movie-genre-distilbert.git](https://github.com/YOUR_GITHUB_USERNAME/movie-genre-distilbert.git)
   cd movie-genre-distilbert
Install requirements:

Bash
pip install -r requirements.txt
Launch the FastAPI Server via Uvicorn:

Bash
uvicorn program1:app --reload
Access the interactive Swagger UI API documentation at http://127.0.0.1:8000/docs.

🧠 Model Performance & Semantic Intelligence
Unlike traditional keyword-matching algorithms (like TF-IDF or SVM), this architecture uses Self-Attention mechanism blocks to infer thematic gravity.

For instance, when evaluating the text:

"Peter asks Dr. Stephen Strange to use magic to make his identity as Spider-Man a secret... the multiverse is broken open and several visitors from alternate realities are brought in."

The network ignores basic keyword lookups and accurately identifies the overlapping contextual layers:

Science Fiction: 88.37% Confidence

Adventure: 83.71% Confidence

Action: 76.00% Confidence
