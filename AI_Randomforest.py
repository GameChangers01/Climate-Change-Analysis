import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("bengaluru.csv")

# Convert the date column to a datetime object
df["date_time"] = pd.to_datetime(df["date_time"])

# Extract features
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["hour"] = df["date_time"].dt.hour

# Define features
X = df[["month", "day", "hour"]]
y = df[["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)

# Create and train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Scale future input data
future_date = pd.to_datetime("2190-10-21")
X_pred = [[future_date.month, future_date.day, future_date.hour]]
X_pred_scaled = scaler_X.transform(X_pred)

# Make predictions
y_pred = model.predict(X_pred_scaled)

# Calculate R-squared
y_test_scaled = scaler_X.transform(X_test)
y_pred_inv = model.predict(y_test_scaled)
r2 = r2_score(y_test, y_pred_inv)
print("R-squared on test data:", r2)

# Print the predictions for the future date
print("Predicted values for the future date:")
for i, col in enumerate(
    ["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]
):
    print(f"{col}: {y_pred[0][i]}")


##43.19%
