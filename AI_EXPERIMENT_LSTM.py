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

# Convert date column to datetime
df["date_time"] = pd.to_datetime(df["date_time"]) 

# Extract features  
df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.day
df["hour"] = df["date_time"].dt.hour

# Define X and y
X = df[["month", "day", "hour"]]  
y = df[["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)

# Create sequences
def create_sequences(X, y, time_steps=3):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 3

X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dense(y_train_seq.shape[1]))

# Compile and train
model.compile(loss='mse', optimizer='adam') 
model.fit(X_train_seq, y_train_seq, epochs=100, verbose=1)

# Create sequence for future date
future_date = pd.to_datetime("2022-05-02")
X_pred = [[future_date.month, future_date.day, future_date.hour]]
X_pred_scaled = scaler_X.transform(X_pred)

# Create dummy y 
dummy_y = np.zeros((1, y_train_seq.shape[1]))

# Create sequence
X_pred_scaled_seq = create_sequences(X_pred_scaled, dummy_y, time_steps) 

# Predict and invert scaling
y_pred_scaled = model.predict(X_pred_scaled_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate on test set
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
y_pred_test_scaled = model.predict(X_test_seq)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)

r2 = r2_score(y_test_seq, y_pred_test)
print("R-squared:", r2)

# Print predictions
print("Predicted values:")
for i, col in enumerate(["maxtempC", "humidity", "cloudcover", "sunHour", "uvIndex", "precipMM"]):
    print(f"{col}: {y_pred[0][i]}")
    
# Plot actual vs predicted temp
plt.plot(y_test["maxtempC"], label="Actual")
plt.plot([None for _ in y_test["maxtempC"]] + [x for x in y_pred[0]], label="Predicted")
plt.legend()
plt.show()