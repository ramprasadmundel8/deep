import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Initialize weights for the two layers
weights1 = np.random.rand(2, 4)
weights2 = np.random.rand(4, 1)
bias1 = np.random.rand(1, 4)
bias2 = np.random.rand(1, 1)

learning_rate = 0.1

for epoch in range(10000):
    # First layer
    hidden_layer_input = np.dot(inputs, weights1) + bias1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Second layer
    final_output = sigmoid(np.dot(hidden_layer_output, weights2) + bias2)
    
    # Backpropagation
    error = expected_output - final_output
    d_predicted_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = d_predicted_output.dot(weights2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    weights2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights1 += inputs.T.dot(d_hidden_layer) * learning_rate
    bias1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch} Loss: {np.mean(np.abs(error))}')

print("Final outputs after training:")
print(final_output)
