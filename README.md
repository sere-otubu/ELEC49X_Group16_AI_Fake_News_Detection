# 🔍 Fake News Detection App

A full-stack application that uses AI (RoBERTa model) to detect fake news and analyze the truthfulness of text content.

## 🏗️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Transformers** - Hugging Face library for NLP models
- **RoBERTa-large-mnli** - Pre-trained transformer model for text classification
- **PyTorch** - Deep learning framework

### Frontend
- **React** - JavaScript UI library
- **Vite** - Fast build tool and dev server
- **Chakra UI** - Modern component library
- **Axios** - HTTP client for API calls

## 📁 Project Structure

```
DeepFake Detection/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── main.jsx        # React entry point
│   ├── index.html          # HTML template
│   ├── package.json        # Node dependencies
│   └── vite.config.js      # Vite configuration
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **4GB+ RAM** (for model loading)

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: First run will download the RoBERTa model (~1.4GB), which may take a few minutes.*

4. **Start the backend server:**
   ```bash
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at: **http://localhost:8000**

5. **Test the API:**
   - Visit http://localhost:8000 for the welcome message
   - Visit http://localhost:8000/docs for interactive API documentation

### Frontend Setup

1. **Open a new terminal and navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The app will open automatically at: **http://localhost:5173**

## 💡 Usage

1. **Start both servers** (backend on port 8000, frontend on port 5173)
2. **Open the web app** in your browser
3. **Enter or paste text** you want to analyze (news article, social media post, etc.)
4. **Click "Analyze Text"** button
5. **View the results:**
   - Truth probability percentage (0-100%)
   - Label: "Likely True" or "Likely False"
   - Visual progress bar
   - Detailed interpretation

## 🔌 API Documentation

### Endpoint: `POST /predict`

**Request:**
```json
{
  "text": "Your text to analyze goes here..."
}
```

**Response:**
```json
{
  "truth_probability": 0.87,
  "label": "true"
}
```

**Fields:**
- `truth_probability`: Float between 0 and 1 (confidence that text is truthful)
- `label`: Either "true" or "false"

### Other Endpoints:
- `GET /` - API info
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🎨 Features

### Backend
- ✅ RoBERTa-large-mnli model for zero-shot classification
- ✅ CORS enabled for local development
- ✅ Error handling and validation
- ✅ Automatic GPU detection (uses CPU if GPU unavailable)
- ✅ Health check endpoint

### Frontend
- ✅ Clean, modern UI with Chakra UI
- ✅ Responsive design
- ✅ Real-time analysis feedback
- ✅ Visual percentage bar
- ✅ Color-coded results (green=true, red=false, yellow=uncertain)
- ✅ Toast notifications
- ✅ Loading states and error handling

## 🧪 Example Texts to Test

**Likely True:**
```
NASA has successfully landed the Perseverance rover on Mars, equipped with advanced instruments to search for signs of ancient microbial life.
```

**Likely False:**
```
Scientists have discovered that drinking coffee can make you invisible for up to 3 hours. The effect was documented in a secret government study.
```

## ⚙️ Configuration

### Change Backend Port
Edit `backend/main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change 8000 to your port
```

### Change Frontend Port
Edit `frontend/vite.config.js`:
```javascript
server: {
  port: 5173  // Change to your preferred port
}
```

Don't forget to update CORS settings in `backend/main.py` if you change the frontend port!

## 🔧 Troubleshooting

### Backend Issues

**Model download fails:**
- Ensure you have a stable internet connection
- Check if you have enough disk space (~2GB free)

**Out of memory error:**
- The model requires significant RAM. Close other applications.
- Consider using `roberta-base` instead of `roberta-large-mnli` in `main.py`

**Port 8000 already in use:**
- Change the port in `main.py` or kill the process using that port

### Frontend Issues

**Cannot connect to backend:**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/main.py`
- Verify the API_URL in `frontend/src/App.jsx`

**npm install fails:**
- Try `npm install --legacy-peer-deps`
- Clear npm cache: `npm cache clean --force`

## 📊 Model Information

The app uses **RoBERTa-large-mnli** (Robustly Optimized BERT Pretraining Approach):
- Zero-shot classification approach
- Trained on Multi-Genre Natural Language Inference (MNLI) dataset
- Can classify text without specific fine-tuning for fake news detection
- Uses hypothesis template: "This text is truthful news" vs "This text is fake news"

**Note:** While powerful, AI models aren't perfect. Always verify important information from multiple reliable sources.

## 🚀 Production Deployment

For production deployment, consider:
- Using a proper database for logging
- Implementing rate limiting
- Adding user authentication
- Caching model predictions
- Using a production WSGI/ASGI server (Gunicorn, Uvicorn with workers)
- Deploying frontend as static build (`npm run build`)
- Setting up proper CORS policies
- Adding monitoring and logging

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Support

If you encounter any issues or have questions, please open an issue on the repository.

---

**Built with ❤️ using React, FastAPI, and RoBERTa**
