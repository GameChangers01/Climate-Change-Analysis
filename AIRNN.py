import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)

# Reshape the data for RNN
X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# Build an RNN model
model = Sequential()
model.add(
    SimpleRNN(
        32, activation="relu", input_shape=(1, X_train.shape[1]), return_sequences=True
    )
)
model.add(SimpleRNN(64, activation="relu"))
model.add(Dense(6))  # 6 output neurons for your 6 target variables

# Compile the model
model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

# Train the model
model.fit(X_train_rnn, y_train, epochs=100, verbose=1)

# Scale future input data
future_date = pd.to_datetime("2045-10-21")
X_pred = [[future_date.month, future_date.day, future_date.hour]]
X_pred_scaled = scaler_X.transform(X_pred)
X_pred_scaled_rnn = X_pred_scaled.reshape(1, 1, X_train.shape[1])

# Make predictions
y_pred_scaled = model.predict(X_pred_scaled_rnn)

# Inverse transform the predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate R-squared
y_test_scaled = scaler_X.transform(X_test)
y_pred_inv = model.predict(y_test_scaled.reshape(X_test.shape[0], 1, X_train.shape[1]))
r2 = r2_score(y_test, y_pred_inv)
print("R-squared on test data:", r2)

# Print the predictions for the future date
print("Predicted values for the future date:")
for i, col in enumerate(
    ["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]
):
    print(f"{col}: {y_pred[0, i]}")


# low than cnn
