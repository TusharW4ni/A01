!pip install --upgrade ucimlrepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, r2_score


# Fetch the dataset
auto_mpg = fetch_ucirepo(id=9)

# Get features and targets
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Create a DataFrame from features and targets
data = X.copy()
data['mpg'] = y

# Handle missing values
data.dropna(subset=['horsepower'], inplace=True)

# Drop unnecessary columns --> model and origin are dropped, by default the car name is not inluded. 
columns_to_drop = [ 'model_year', 'origin']  
data.drop(columns=columns_to_drop, inplace=True)

# Normalize/scale the data
scaler = StandardScaler()
scaled_data_array = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data_array, columns=data.columns)

# Split into features and target variable
X = scaled_data.drop('mpg', axis=1).values
y = scaled_data['mpg'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80/20 split

# Add a column of ones to include the intercept in the model --> bias 
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Initialize parameters for gradient descent
alpha = 0.2     # Learning rate --> variable learning rates used. such as 0.0001, 0.5, 1, etc
iterations = 200  # Number of iterations  --> varaible iterations used such as 100, 1500, 1000, 10000 
tolerance = 1e-6  # Tolerance for convergence

print(f"Learning Rate: {alpha}")
print(f"Number of Iterations: {iterations}")
print(f"Tolerance: {tolerance}")

# Function to compute the Mean Squared Error
def compute_mse(x, y, theta):
    predictions = x.dot(theta)
    errors = predictions - y
    mse = np.mean(errors ** 2)
    return mse

# Define the cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Implement gradient descent
def gradient_descent(X_train, y_train, X_test, y_test, theta, learning_rate, n_iterations, tolerance):
    m_train = len(y_train)
    m_test = len(y_test)
    cost_history_train = []
    cost_history_test = []
    for i in range(n_iterations):
        gradients = (1 / m_train) * X_train.T.dot(X_train.dot(theta) - y_train)
        theta = theta - learning_rate * gradients
        cost_train = compute_cost(X_train, y_train, theta)
        cost_history_train.append(cost_train)
        # Calculate cost on test set
        cost_test = compute_cost(X_test, y_test, theta)
        cost_history_test.append(cost_test)
        if i > 0 and abs(cost_history_train[-2] - cost_history_train[-1]) < tolerance:
            print(f"Converged after {i} iterations")
            break
    return theta, cost_history_train, cost_history_test

# Initialize theta for linear regression
theta = np.zeros(X_train.shape[1])

# Perform gradient descent on scaled data
theta, mse_history_train, mse_history_test = gradient_descent(X_train, y_train, X_test, y_test, theta, alpha, iterations, tolerance)
# Print final theta values --> these are the weight coeff
print("Final Theta values after gradient descent:")
print(theta)

# Compute MSE on training set
train_predictions = X_train.dot(theta)
train_mse = compute_mse(X_train, y_train, theta)
print(f"MSE on training set: {train_mse}")

# Compute and print explained variance and R² scores for training set 
explained_variance_training = explained_variance_score(y_train, train_predictions)
r2_training = r2_score(y_train, train_predictions)
print(f"Explained Variance on training set: {explained_variance_training}")
print(f"R² on training set: {r2_training}\n")

# Compute predictions on the test set
test_predictions = X_test.dot(theta)

# Compute and print the final MSE for test set
test_mse = compute_mse(X_test, y_test, theta)
print(f"Final MSE on test set: {test_mse}")

# Compute and print explained variance and R² scores for test set
explained_variance_test = explained_variance_score(y_test, test_predictions)
r2_test = r2_score(y_test, test_predictions)
print(f"Explained Variance on test set: {explained_variance_test}")
print(f"R² on test set: {r2_test}")



# Plot MSE vs. Iterations
print(" ")
print("Graph to show MSE vs Iterations for training set: ")
plt.figure(figsize=(5, 3))
iterations = range(1, len(mse_history_train) + 1)
plt.plot(iterations, mse_history_train, marker='o', linestyle='-', color='b', label='Training MSE')
plt.plot(iterations, mse_history_test, marker='o', linestyle='-', color='orange', label='Test MSE')
plt.title('MSE vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Function to plot Output Variable (MPG) vs. Important Attribute (Horsepower) --> only needed to printed once. same data as part 2 since we are using the same values 
def plot_output_vs_attribute(attribute_values, output_values):
    plt.figure(figsize=(7, 4))
    plt.scatter(attribute_values, output_values, color='b', alpha=0.6)
    plt.title('MPG vs. Horsepower')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.grid(True)
    plt.show()

# Plot Output Variable (MPG) vs. Horsepower for test set
horsepower_test = X_test[:, 1]  # Assuming 'horsepower' is the first column after intercept
plot_output_vs_attribute(horsepower_test, test_predictions)
