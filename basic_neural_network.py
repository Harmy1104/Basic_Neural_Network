"""
Basic prediction neural network. If the first number is 1 then output is the second number else third number
"""

# Importing numpy --------------------------------------
import numpy as np

# Function definations ---------------------------------
# f(x) = sigmoid function {1 / (1 + (e^-x))}
def f(x):
	return 1 / (1 + np.exp(-x))

# For backpropogation
def df_dx(x):
	return x * (1 - x)

# Defining training data -------------------------------
training_data = np.array([[1, 0, 0],
						  [1, 0, 1], 
						  [1, 1, 0],
						  [0, 0, 0],
						  [0, 0, 1],
						  [0, 1, 0]])

# Real outputs of the training data --------------------
outputs = np.array([[0, 0, 1, 0, 1, 0]]).T  # array of arrays([[1, 1, 0, 0]]) // not([1, 1, 0, 0]) 

# Defining Weights and epochs --------------------------
weight = 2 * np.random.rand(3, 1) 
epochs = 100000

# Actual training part ---------------------------------
for _ in range(epochs):
	
	input_layer = training_data
	
	training_output = f(np.dot(input_layer, weight))

	# Calculating the error between the real outputs and the training outputs for adjusting the weights
	error = outputs - training_output

	# Adjustment values i.e. sum between error and training_outputs passed through df_dx() (derivation of the sigmoid function)
	adjustments = error + df_dx(training_output)

	# Actually adjusting the weights
	# new weights = weights + adjustments
	weight += np.dot(input_layer.T, adjustments)

# Predicts the output ----------------------------------
def predict(x):
	print(f(np.dot(x, weight)))

if __name__ == "__main__":
	predict([0, 1, 1]) 
