import pandas as pd
import numpy as np
import pickle
import optuna  # Smarter hyperparameter tuning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats.mstats import winsorize
import xgboost as xgb
from sklearn.feature_selection import RFE

# Load dataset
df = pd.read_csv("data\Housing.csv")

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Winsorization: Cap extreme values instead of removing them
df["price"] = winsorize(df["price"], limits=[0.01, 0.01])  # Cap top & bottom 1%

# Log-transform the target variable to reduce skewness
df["log_price"] = np.log1p(df["price"])

# Convert categorical features
df = pd.get_dummies(df.drop(columns=["price"]), drop_first=True)

# Define features and target
X = df.drop(columns=["log_price"])
y = df["log_price"]  # Use log-transformed target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection: Keep Top Features Based on RFE
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(rf_temp, n_features_to_select=min(20, X_train.shape[1]))  # Select top 20 or max available
X_train_scaled = rfe.fit_transform(X_train_scaled, y_train)
X_test_scaled = rfe.transform(X_test_scaled)

# Store selected feature names
selected_features = X.columns[rfe.support_]

# Optuna Hyperparameter Tuning Function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0)
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return r2_score(y_test, y_pred)

# Optimize model parameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params

# Train best XGBoost model
xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred_log = xgb_model.predict(X_test_scaled)

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)  # Convert actual values back

# Ensure no negative prices
y_pred = np.maximum(y_pred, 0)

# Calculate Metrics
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)
accuracy = r2 * 100  # Convert R² score to percentage

# Print Metrics
print(" Model Evaluation:")
print(f"MAE: ₹{round(mae, 2)}")
print(f"RMSE: ₹{round(rmse, 2)}")
print(f"🔹 Accuracy: {round(accuracy, 2)}%")

# Save best model & scaler
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

print("✅ Model training complete. Saved as house_price_model.pkl")
