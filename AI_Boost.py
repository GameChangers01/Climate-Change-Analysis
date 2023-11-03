# Imports
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("bengaluru.csv")

# Extract features

df["date_time"] = pd.to_datetime(df["date_time"])

# Extract features
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["hour"] = df["date_time"].dt.hour

X = df[["month", "day", "hour"]]
y = df["maxtempC"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# Make a prediction on new data
future_date = "2190-10-21"
X_new = pd.DataFrame({"month": [10], "day": [21], "hour": [0]})

# Scale and predicts
X_new = scaler.transform(X_new)
y_new_pred = model.predict(X_new)
print("New date prediction:", y_new_pred)
