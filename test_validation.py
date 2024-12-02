import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
file_path = 'Your Dataset.xlsx'
dataset = pd.read_excel(file_path)

# Select Features and Target
features = dataset[['Your Parameters', 'Your Parameters', 'Your Parameters', 'Your Parameters']]
target = dataset['Target Parameters']

# Define the model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# List of test sizes and validation splits to test
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
validation_splits = [0.1, 0.2, 0.3]  # Fraction of the training data to be used for validation

# Dictionary to store the results
results = {}

for test_size in test_sizes:
    for val_split in validation_splits:
        print(f"Testing test_size: {test_size}, validation_split: {val_split}")
        
        # Split dataset into training set and test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
        
        # Further split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, random_state=42)
        
        # Create a new model for each configuration
        model = create_model()
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=0)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)
        
        # Store the result
        results[(test_size, val_split)] = mse
        print(f"MSE for test_size {test_size}, validation_split {val_split}: {mse:.2f}\n")

# Plot the results
plt.figure(figsize=(10,6))
for val_split in validation_splits:
    test_size_mse = [(test_size, mse) for (test_size, vs), mse in results.items() if vs == val_split]
    test_sizes_plot, mse_values = zip(*test_size_mse)
    plt.plot(test_sizes_plot, mse_values, marker='o', label=f'Validation Split: {val_split}')

plt.title('Test Size vs Mean Squared Error (MSE) for Different Validation Splits')
plt.xlabel('Test Size')
plt.ylabel('MSE Value')
plt.legend()
plt.grid(True)
plt.show()

# Find and print the best configuration (based on the lowest MSE)
best_config = min(results, key=results.get)
print(f"\nBest Configuration (based on lowest MSE):")
print(f"Test Size: {best_config[0]}, Validation Split: {best_config[1]}")
print(f"Best MSE: {results[best_config]:.2f}")
