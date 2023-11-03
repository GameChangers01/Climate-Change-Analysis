import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("bengaluru.csv")

# Convert the date column to a datetime object
df["date_time"] = pd.to_datetime(df["date_time"])

# Extract features
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["year"] = df["date_time"].dt.year
df["hour"] = df["date_time"].dt.hour

# Define features
X = df[["year", "month", "day", "hour"]]
y = df[["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)

# Create and train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Set the desired future year
desired_year = 2599  # Change this to any year you want to predict

# Create the input data for prediction
X_pred = [[desired_year, 10, 21, 0]]
X_pred_scaled = scaler_X.transform(X_pred)

# Make predictions
y_pred = model.predict(X_pred_scaled)

# Calculate R-squared (R2) for each attribute
y_test_scaled = scaler_X.transform(X_test)
y_pred_inv = model.predict(X_test)
r2_scores = [
    r2_score(y_test[col], y_pred_inv[:, idx]) for idx, col in enumerate(y.columns)
]

# Print the predictions for the desired year
print(f"Predicted climate attributes for the year {desired_year}:")
for i, col in enumerate(y.columns):
    print(f"{col}: {y_pred[0][i]}")

# Print accuracy percentages for each attribute
print("Accuracy percentages for each attribute:")
for idx, col in enumerate(y.columns):
    accuracy_percentage = max(0, (1 - r2_scores[idx]) * 100)
    print(f"{col}: {accuracy_percentage:.2f}%")


# Utt  FLOP ---Bhavish
