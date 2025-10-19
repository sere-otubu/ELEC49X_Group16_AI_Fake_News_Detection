# Backend - Fake News Detection API

FastAPI backend with RoBERTa model for fake news detection.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## API Endpoints

### POST /predict
Analyze text for truthfulness.

**Request:**
```json
{
  "text": "Your news text here..."
}
```

**Response:**
```json
{
  "truth_probability": 0.87,
  "label": "true"
}
```

### GET /
API information

### GET /health
Health check endpoint

### GET /docs
Interactive API documentation (Swagger UI)

## Model

Uses **RoBERTa-large-mnli** for zero-shot classification:
- Classifies text as "truthful news" vs "fake news"
- Returns probability score (0-1)
- Automatically uses GPU if available, otherwise CPU

## Development

```bash
# Install in development mode
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --port 8000
```

## Testing

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test text here"}'
```

Or visit http://localhost:8000/docs for interactive testing.
