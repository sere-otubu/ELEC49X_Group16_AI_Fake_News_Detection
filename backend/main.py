from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
logger.info("Loading RoBERTa model...")
try:
    # Using zero-shot classification with RoBERTa for fake news detection
    classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    classifier = None

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    truth_probability: float
    label: str

@app.get("/")
def read_root():
    return {
        "message": "Fake News Detection API",
        "status": "running",
        "model": "roberta-large-mnli"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput):
    """
    Predict whether the given text is true or fake news.
    
    Returns:
        - truth_probability: Float between 0 and 1 (0 = likely false, 1 = likely true)
        - label: "true" or "false"
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not input_data.text or len(input_data.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Use zero-shot classification with candidate labels
        result = classifier(
            input_data.text,
            candidate_labels=["truthful news", "fake news"],
            hypothesis_template="This text is {}."
        )
        
        # Extract scores
        labels = result['labels']
        scores = result['scores']
        
        # Find the score for "truthful news"
        truth_index = labels.index("truthful news")
        truth_probability = scores[truth_index]
        
        # Determine label
        label = "true" if truth_probability > 0.5 else "false"
        
        return PredictionOutput(
            truth_probability=round(truth_probability, 4),
            label=label
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
