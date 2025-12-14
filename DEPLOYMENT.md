# 🚀 Deployment Guide

## Quick Deploy Options

### Option 1: Railway (Backend) + Streamlit Cloud (Frontend) - RECOMMENDED

#### Backend on Railway:
1. Push to GitHub
2. Connect Railway to your repo
3. Railway auto-detects and deploys
4. Get your API URL: `https://your-app.railway.app`

#### Frontend on Streamlit Cloud:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set main file: `streamlit_app.py`
4. Add environment variable: `API_BASE_URL = https://your-app.railway.app`
5. Deploy!

### Option 2: Railway Only (Both Services)

1. Deploy backend first (main branch)
2. Create separate Railway service for frontend
3. Set environment variables

## Environment Variables

### Backend (Railway):
- `ENVIRONMENT=production` (optional - disables docs)
- `PORT` (auto-set by Railway)

### Frontend (Streamlit Cloud):
- `API_BASE_URL=https://your-backend-url.railway.app`

## Files Structure for Deployment:

```
├── app.py                 # Railway backend entry point
├── streamlit_app.py       # Streamlit Cloud entry point  
├── backend/               # Backend modules
├── frontend/              # Frontend modules
├── requirements.txt       # All dependencies
├── Procfile              # Railway config
├── runtime.txt           # Python version
└── README.md             # Documentation
```

## Testing Deployment Locally:

### Backend:
```bash
python app.py
# Visit: http://localhost:8000
```

### Frontend:
```bash
streamlit run streamlit_app.py
# Visit: http://localhost:8501
```

## Deployment Checklist:

- ✅ All imports are absolute (not relative)
- ✅ Models are in backend/models/
- ✅ x_test.csv is in backend/
- ✅ Requirements.txt includes all dependencies
- ✅ Environment variables are set
- ✅ .gitignore excludes unnecessary files

## Post-Deployment:

1. Test all API endpoints
2. Test frontend prediction functionality
3. Test feature importance charts
4. Monitor logs for any issues

## Troubleshooting:

- **Import errors**: Check Python path in entry points
- **Model loading fails**: Ensure models/ folder is included
- **API connection fails**: Check API_BASE_URL environment variable
- **Memory issues**: Railway free tier has 512MB limit