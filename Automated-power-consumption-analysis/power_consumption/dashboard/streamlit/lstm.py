# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, LSTM, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.optimizers import Adam

# # Load and preprocess the data
# def load_and_preprocess_data(file_path):
#     # Try multiple encodings
#     encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
#     for encoding in encodings:
#         try:
#             # Read the CSV file with explicit date parsing and specified encoding
#             df = pd.read_csv(file_path, parse_dates=['Timestamps'], dayfirst=True, encoding=encoding)
            
#             # Print first few rows to verify data
#             print(f"Successfully read file with {encoding} encoding")
#             print(df.head())
            
#             break
#         except UnicodeDecodeError:
#             continue
#     else:
#         raise ValueError("Could not read the file with any of the specified encodings")
    
#     # Sort by timestamp to ensure chronological order
#     df = df.sort_values('Timestamps')
    
#     # Remove any * characters from numerical columns
#     df['ampere'] = df['ampere'].astype(str).str.replace('*', '').astype(float)
#     df['wattage_kwh'] = df['wattage_kwh'].astype(str).str.replace('*', '').astype(float)
#     df['pf'] = df['pf'].astype(str).str.replace('*', '').astype(float)
    
#     # Select features for prediction
#     features = ['ampere', 'wattage_kwh', 'pf']
    
#     return df, features

# # Prepare sequences for LSTM
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i+seq_length])
#         y.append(data[i+seq_length])
    
#     return np.array(X), np.array(y)

# # Build LSTM model with Input layer
# def build_lstm_model(input_shape, output_shape):
#     # Create Input layer
#     inputs = Input(shape=input_shape)
    
#     # LSTM layers
#     x = LSTM(50, activation='relu', return_sequences=True)(inputs)
#     x = LSTM(50, activation='relu')(x)
    
#     # Output layer
#     outputs = Dense(output_shape)(x)
    
#     # Create the model
#     model = Model(inputs=inputs, outputs=outputs)
    
#     # Compile the model
#     model.compile(
#         optimizer=Adam(learning_rate=0.001), 
#         loss=MeanSquaredError(),
#         metrics=['mse']  # Optional: add mean absolute error as a metric
#     )
    
#     return model

# # Main prediction pipeline
# def predict_time_series(file_path, seq_length=3, epochs=50, future_steps=5):
#     # Load and preprocess data
#     df, features = load_and_preprocess_data(file_path)
    
#     # Normalize the features
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df[features])
    
#     # Create sequences
#     X, y = create_sequences(scaled_data, seq_length)
    
#     # Flatten the input for LSTM
#     X = X.reshape(X.shape[0], seq_length, len(features))
    
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     # Build and train the model
#     model = build_lstm_model(
#         input_shape=(seq_length, len(features)), 
#         output_shape=len(features)
#     )
    
#     # Fit the model
#     history = model.fit(
#         X_train, y_train, 
#         epochs=epochs, 
#         batch_size=32, 
#         validation_split=0.2, 
#         verbose=1
#     )
    
#     # Evaluate the model
#     test_loss = model.evaluate(X_test, y_test)
#     print(f"Test Loss: {test_loss}")
    
#     # Predict future values
#     last_sequence = X[-1]  # Start with the last sequence from the test data
#     future_predictions = []

#     for _ in range(future_steps):
#         # Predict the next time step
#         next_pred = model.predict(last_sequence.reshape(1, seq_length, len(features)), verbose=0)
        
#         # Add prediction to the future predictions
#         future_predictions.append(next_pred[0])
        
#         # Update the sequence by adding the prediction and removing the oldest time step
#         last_sequence = np.vstack([last_sequence[1:], next_pred[0]])

#     # Inverse transform the predictions
#     future_predictions = scaler.inverse_transform(np.array(future_predictions))

#     # Create a DataFrame with predictions
#     future_timestamps = pd.date_range(
#         start=df['Timestamps'].iloc[-1], 
#         periods=future_steps + 1, 
#         freq='5T'
#     )[1:]

#     predictions_df = pd.DataFrame(
#         future_predictions, 
#         columns=features, 
#         index=future_timestamps
#     )

#     return predictions_df, model, scaler

# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual file path
#     file_path = r'C:\Users\Dell\Desktop\new123.csv'
    
#     # Predict future values
#     future_predictions, trained_model, scaler = predict_time_series(file_path)
    
#     # Print the predictions
#     print("\nFuture Predictions:")
#     print(future_predictions)
    
#     # Save the model with custom configuration
#     trained_model.save('electrical_measurements_lstm_model.h5', save_format='tf')
    
#     # Save the scaler
#     import joblib
#     joblib.dump(scaler, 'scaler.joblib')

#     # Optional: Plot actual vs predicted values
#     try:
#         import matplotlib.pyplot as plt
        
#         plt.figure(figsize=(15,5))
#         for i, feature in enumerate(['ampere', 'wattage_kwh', 'pf']):
#             plt.subplot(1,3,i+1)
#             plt.title(f'Predictions for {feature}')
#             plt.plot(future_predictions.index, future_predictions[feature], label='Predicted')
#             plt.xlabel('Timestamp')
#             plt.ylabel(feature)
#             plt.legend()
#         plt.tight_layout()
#         plt.show()
#     except Exception as e:
#         print("Could not generate plot:", e)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import sys

# Ensure console output is UTF-8 encoded (fix for UnicodeEncodeError)
sys.stdout.reconfigure(encoding='utf-8')

# Load the dataset from the CSV file
file_path = r"C:\Users\Dell\Desktop\new123.csv"  # Replace this with your CSV file path
df = pd.read_csv(file_path)

# Ensure the timestamps column is properly parsed
df['Timestamps'] = pd.to_datetime(df['Timestamps'], dayfirst=True)

# Normalize the features (ampere, wattage_kwh, pf)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['ampere', 'wattage_kwh', 'pf']])

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 3  # Use the last 3 timesteps to predict the next
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 3), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(3)  # Predicting 3 values: ampere, wattage_kwh, pf
])

model.compile(optimizer='adam',loss=MeanSquaredError(), metrics=['mse'])

# Train the model
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = model.fit(X, y, epochs=200, batch_size=8, verbose=2, callbacks=[early_stopping])  # Suppressed verbose output

# Save the model to an H5 file
model_path = "future_values_prediction_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

# Save the scaler for future use
import joblib
scaler_path = "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")
