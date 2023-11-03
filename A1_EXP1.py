# Imports 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

# Load data
df = pd.read_csv("bengaluru.csv")

# Extract year as a feature  
df["date_time"] = pd.to_datetime(df["date_time"])

# Now extract year 
df["year"] = df["date_time"].dt.year
X = df[["year"]] 
y = df["maxtempC"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2*100)

# Make a prediction on a future year
future_year = 2190
X_new = pd.DataFrame({"year": [future_year]})

# Scale and predict
X_new = scaler.transform(X_new)  
y_new_pred = model.predict(X_new)
print("Prediction for year", future_year, ":", y_new_pred)