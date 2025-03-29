# House Price Prediction API

## Overview
This project is a **House Price Prediction API** built using **Flask** and **Machine Learning**. It takes house attributes as input and predicts the price in **INR**.

## Features
- **Predict house prices** based on key attributes
- **Supports JSON input** via REST API
- **Handles missing values** automatically
- **Optimized model** with hyperparameter tuning
- **Deployed on Render** for live access

##  Model & Data
- **Dataset**: Publicly available house price dataset (e.g., Kaggle, California Housing)
- **Preprocessing**: Missing values, scaling, feature selection, encoding
- **Algorithm Used**: Random Forest Regressor (optimized with GridSearchCV)
- **Evaluation Metrics**: RMSE, MAE, R² Score

##  Tech Stack
- **Python** (pandas, numpy, scikit-learn, flask)
- **Flask** (for API)
- **Pickle** (for model serialization)
- **Render** (for deployment)

##  API Endpoints
### 1️ **Welcome Message**
```
GET /
```
**Response:**
```json
{"message": "Welcome to the House Price Prediction API!"}
```

### 2️ **Predict House Price**
```
POST /predict
```
**Request Body (JSON Example):**
```json
{
    "area": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    "furnishing": "furnished",
    "air_conditioning": "yes"
}
```

**Response:**
```json
{
    "predicted_price": "₹3,478,201"
}
```

##  Installation & Usage
1️ **Clone the Repository**
```sh
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

2️ **Install Dependencies**
```sh
pip install -r requirements.txt
```

3️ **Run Flask API**
```sh
python app.py
```

4️ **Test in Postman or cURL**
```sh
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"area":1200, "bedrooms":3, "bathrooms":2, "stories":2, "parking":1, "furnishing":"furnished", "air_conditioning":"yes"}'
```

##  Deployment on Render
- Hosted at: **[Live API URL](https://house-price-app.onrender.com)**
- GitHub Actions configured for auto-deployment



