# Test Size and Validation Split Analysis in Neural Networks


 
## Overview
This code evaluates the impact of different **test set sizes** and **validation splits** on the performance of a simple neural network for regression tasks. 
The objective is to identify the best combination of test size and validation split that minimizes the **Mean Squared Error (MSE)** on the test set.

---

## How the Code Works

### 1. Dataset Loading
- The dataset is loaded from an external file into a pandas DataFrame.
- The dataset should contain:
  - **Features**: Independent variables used for predictions.
  - **Target**: Dependent variable to be predicted.

### 2. Model Definition
- A simple feedforward neural network is created using TensorFlow's Keras API.
  - **Input Layer**: Matches the number of features in the dataset.
  - **Hidden Layers**: Two hidden layers with 64 and 32 neurons, respectively, using ReLU activation.
  - **Output Layer**: A single neuron for regression output.
- The model uses the Adam optimizer and Mean Squared Error (MSE) as the loss function.

### 3. Configurations
- **Test Sizes**: Proportions of the dataset reserved for testing (e.g., `0.1`, `0.2`, etc.).
- **Validation Splits**: Proportions of the training set reserved for validation during training (e.g., `0.1`, `0.2`, etc.).

### 4. Data Splitting
- For each test size:
  - The dataset is split into training and test sets.
  - The training set is further split into training and validation subsets based on the validation split value.
- This ensures the model is trained and evaluated consistently across different configurations.

### 5. Model Training and Evaluation
- For each combination of test size and validation split:
  - A new model is created and trained for 10 epochs.
  - The model's performance is evaluated on the test set using MSE.
- Results are stored and printed for each configuration.

### 6. Visualization
- The results are plotted, showing how test size impacts MSE for each validation split configuration.
- The plot provides insights into the trade-offs between test size and model performance.

### 7. Optimal Configuration
- The best configuration, i.e., the one with the lowest MSE, is identified and printed.

---

## Purpose of the Code
- To analyze the impact of dataset splitting strategies on neural network performance.
- To identify the optimal test size and validation split for a given dataset.
- To provide a reusable framework for evaluating data splitting strategies in regression tasks.

---

## General Use Cases
- Regression tasks where minimizing test set prediction error (MSE) is critical.
- Testing the robustness of models under different data splitting strategies.
- Understanding the trade-offs between training, validation, and test set sizes.

---

## Requirements
- **Input Dataset**:
  - The dataset must include both features and a target variable.
  - Modify the code to specify appropriate feature and target column names.
- **Libraries**:
  - `numpy`, `pandas`, `tensorflow`, `matplotlib`.
- **Hardware**:
  - GPU acceleration is recommended for faster training, especially for large datasets.

---

## Notes
- The code is designed for regression tasks but can be adapted for classification tasks by changing the model architecture and loss function.
- The test size and validation split ranges can be adjusted to suit different datasets or tasks.

---

This modular code structure allows users to analyze the effects of different dataset splits on model performance and choose the best configuration for their specific use case.