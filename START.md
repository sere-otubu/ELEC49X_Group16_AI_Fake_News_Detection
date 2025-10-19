# 🚀 Quick Start Guide

## First Time Setup

### 1. Backend Setup (Terminal 1)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Note:** First run downloads the RoBERTa model (~1.4GB). This may take 5-10 minutes depending on your internet speed.

### 2. Frontend Setup (Terminal 2)

```bash
cd frontend
npm install
npm run dev
```

---

## Daily Usage

After the first setup, you only need:

### Terminal 1 - Backend
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

---

## Access Points

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## Troubleshooting

**Port already in use?**
- Backend: Change port in `backend/main.py`
- Frontend: Change port in `frontend/vite.config.js`

**Model not loading?**
- Check internet connection for first-time download
- Ensure you have ~2GB free disk space
- Try restarting the backend

**Frontend can't connect?**
- Make sure backend is running
- Check if backend is on port 8000
- Check browser console for errors

---

**Need more help?** Check the main README.md file!
