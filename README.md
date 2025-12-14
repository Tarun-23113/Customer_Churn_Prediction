# 🚀 Customer Churn Prediction - Full Stack ML Application

A production-ready machine learning application for predicting customer churn with a FastAPI backend and Streamlit frontend.

## 🏗️ Architecture

```
Frontend (Streamlit) ←→ REST API ←→ Backend (FastAPI)
     │                                    │
   User Interface              ML Models + Analytics
```

## ✨ Features

### 🔮 Predictions Page
- Real-time churn predictions
- Multiple ML models (Random Forest, XGBoost, Gradient Boosting, Logistic Regression)
- Interactive feature inputs with proper validation
- Confidence scoring and probability visualization
- Customizable prediction threshold

### 📈 Insights Page
- Feature importance analysis
- Model performance comparison
- Interactive charts and visualizations
- Model interpretability tools

## 🚀 Quick Start

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
API will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

### Frontend (Streamlit)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
App will be available at: http://localhost:8501

## 📁 Project Structure

```
churn-prediction/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models/              # Trained ML models (.pkl files)
│   ├── x_test.csv          # Sample data for feature ranges
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit application
│   └── requirements.txt
├── data/                   # Training data (optional)
└── README.md
```

## 🔧 API Endpoints

- `GET /` - API status and info
- `GET /health` - Health check
- `GET /models` - Available models
- `GET /feature-stats` - Feature statistics
- `POST /predict/{model_name}` - Make predictions
- `GET /feature-importance/{model_name}` - Get feature importance

## 🚀 Optimized Deployment

### Performance Optimizations:
- **Memory efficient**: Models loaded once, cached predictions
- **Fast startup**: Optimized Docker images with multi-stage builds
- **Compressed responses**: GZip middleware enabled
- **Caching**: LRU cache for feature importance, API responses
- **Input validation**: Pydantic models with constraints
- **Resource limits**: Docker memory limits for cost efficiency

### Deployment Options:

**Option 1: Railway (Recommended)**
1. Fork this repository
2. Connect to Railway
3. Deploy backend: `railway up` (uses railway.json)
4. Deploy frontend: Set `API_BASE_URL` environment variable

**Option 2: Docker Compose (Local/VPS)**
```bash
docker-compose up --build
```

**Option 3: Separate Services**
- Backend: Railway/Render/Fly.io
- Frontend: Streamlit Cloud

### Environment Variables:
- `API_BASE_URL`: Backend URL (for frontend)
- `ENVIRONMENT`: Set to "production" to disable docs
- `PORT`: Port for Railway deployment (auto-set)

## 🛠️ Technologies Used

**Backend:**
- FastAPI - Modern, fast web framework
- Pydantic - Data validation
- Scikit-learn - ML models
- Pandas - Data manipulation
- Uvicorn - ASGI server

**Frontend:**
- Streamlit - Web app framework
- Plotly - Interactive visualizations
- Requests - HTTP client

## 📊 Models Included

1. **Random Forest** - Ensemble method, good for feature importance
2. **XGBoost** - Gradient boosting, high performance
3. **Gradient Boosting** - Sequential ensemble method
4. **Logistic Regression** - Linear model, interpretable

## 🎯 Features for Prediction

- Price
- Freight Value
- Payment Installments
- Delivery Difference (days)
- Reviewed Days
- Customer State (encoded)
- Product Category (encoded)
- Payment Type (encoded)

## 🔮 Future Enhancements

- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Real-time monitoring
- [ ] User authentication
- [ ] Model versioning
- [ ] Batch predictions
- [ ] Data drift detection

## 📈 Performance

- **API Response Time**: < 100ms
- **Model Loading**: On startup (faster predictions)
- **Concurrent Users**: Supports multiple simultaneous requests
- **Scalability**: Horizontal scaling ready

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Built with ❤️ for the ML community**