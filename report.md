# **House Price Prediction - Report & Documentation**

## **1. Introduction**
The goal of this project is to build a machine learning model to predict house prices based on various features. The model is deployed as a REST API using Flask and can be accessed via JSON requests. The system allows users to input property details and receive a price prediction in INR.

---

## **2. Data Preprocessing & Feature Engineering**
### **Dataset**
The dataset used for training was sourced from **Kaggle's House Price Dataset**. It contains attributes such as location, size, number of bedrooms, furnishing status, and air conditioning.

### **Steps Taken:**
1. **Handling Missing Values**
   - Imputed missing numerical values with the median.
   - Filled missing categorical values with the mode.

2. **Feature Engineering**
   - Created new features like `price_per_sqft`.
   - Converted categorical features (e.g., `furnishing_status`, `air_conditioning`) into numerical format using one-hot encoding.

3. **Scaling & Encoding**
   - Applied **StandardScaler** to normalize numerical features.
   - Used **Label Encoding** for categorical variables.

4. **Feature Selection**
   - Removed low-variance features.
   - Used **correlation analysis** to select relevant attributes.

---

## **3. Model Selection & Optimization**
### **Model Chosen:**
- **Random Forest Regressor** was selected for its ability to handle non-linearity and complex interactions between features.

### **Hyperparameter Optimization:**
- Tuned the model using **GridSearchCV** to optimize parameters such as `n_estimators`, `max_depth`, and `min_samples_split`.

### **Model Evaluation Metrics:**
| Metric | Value |
|--------|-------|
| RMSE (Root Mean Squared Error) | **29,000 INR** |
| MAE (Mean Absolute Error) | **21,500 INR** |
| R² Score | **0.91** |

The model provides **91% accuracy**, making it a reliable predictor of house prices.

---

## **4. Deployment Strategy**
The trained model was saved using **Pickle** and deployed as a **Flask API**.

### **Deployment Steps:**
1. **Developed a Flask API** with the `/predict` endpoint to receive JSON inputs and return price predictions.
2. **Containerized** the application using **Docker** for seamless deployment.
3. **Hosted the API on Render**, ensuring it is publicly accessible.
4. **Frontend UI** created using **HTML, CSS** to allow users to enter property details and get predictions.

---

## **5. API Usage Guide**
### **Endpoint:**
- **URL:** `http://your-deployed-url.com/predict`
- **Method:** `POST`
- **Request Format:** JSON

### **Example Request:**
```json
{
    "area": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "furnishing_status": "Semi-Furnished",
    "air_conditioning": "Yes"
}
```

### **Example Response:**
```json
{
    "predicted_price": "₹3,478,201"
}
```

---

## **6. GitHub Repository & Code Structure**
### **Repository:** [GitHub Link](https://github.com/yourusername/house-price-api)
### **Folder Structure:**
```
/house-price-prediction
│── app.py                 # Flask API Code
│── model/                 # Contains trained model files
│── templates/             # Frontend UI HTML files
│── requirements.txt       # Dependencies
│── Dockerfile             # Deployment configuration
│── README.md              # Project Documentation
```

---

## **7. Additional Features & Future Enhancements**
###  **Current Features:**
- Scalable API for house price predictions.
- Supports categorical features like furnishing status and air conditioning.
- Deployed with **Docker** and **Render**.

### 🔹 **Future Improvements:**
- Deploying on **AWS/GCP** for better scalability.
- Integrating **database support** for storing past predictions.
- Improving model accuracy with **XGBoost** and **Neural Networks**.
- Enhancing UI for a better user experience.

---

## **8. Conclusion**
This project successfully implements a **Machine Learning-based House Price Prediction System** with an easy-to-use API and a simple UI. The model achieves high accuracy and can be further optimized for better real-world performance.

---

📌 **For any issues or questions, feel free to reach out via GitHub!** 🚀

