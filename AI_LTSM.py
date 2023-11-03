import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("bengaluru.csv")
tf.config.run_functions_eagerly(True)
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)


# Convert data to sequences for LSTM
def create_sequences(X, y, time_steps=3):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


time_steps = 3  # You can adjust this based on your needs
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)

# Build an LSTM model
model = Sequential()
model.add(
    LSTM(
        64, activation="relu", input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
    )
)
model.add(Dense(y_train_seq.shape[1]))

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Train the model
model.fit(X_train_seq, y_train_seq, epochs=100, verbose=1)

# Scale and prepare future input data
future_date = pd.to_datetime("2022-05-02")
X_pred = [[future_date.month, future_date.day, future_date.hour]]

X_pred_scaled = scaler_X.transform(X_pred)
X_pred_scaled_seq = create_sequences(
    X_pred_scaled, np.zeros((1, y_train_seq.shape[1]))
)[0]

# Make predictions
y_pred_scaled = model.predict(X_pred_scaled_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate R-squared on the test data
X_test_seq, y_test_seq = create_sequences(
    scaler_X.transform(X_test), scaler_y.transform(y_test), time_steps
)
y_pred_test_scaled = model.predict(X_test_seq)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)

r2 = r2_score(y_test_seq, y_pred_test_scaled)
print("R-squared on test data:", r2)

# Print the predictions for the future date
print("Predicted values for the future date:")
for i, col in enumerate(
    ["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]
):
    print(f"{col}: {y_pred[0][i]}")

# Plot actual vs. predicted values for temperature
plt.figure(figsize=(12, 6))
plt.plot(y_test["maxtempC"].values, label="Actual", marker="o")
plt.plot(
    [None for _ in y_test["maxtempC"]] + [x for x in y_pred[0]],
    label="Predicted",
    marker="o",
)
plt.title("Actual vs. Predicted Temperature (maxtempC)")
plt.legend()
plt.show()
