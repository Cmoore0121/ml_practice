import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" Neural Network with 2 layers """



data = pd.read_csv("DigitRecognition/digits28x28.csv").to_numpy()
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

X = train_data[:, 1:] / 255.0
Y = train_data[:, 0]
X_test = test_data[:, 1:] / 255.0
Y_test = test_data[:, 0]

input_size = 784
hidden_size1 = 64
hidden_size2 = 32
output_size = 10
learning_rate = 0.04
epochs = 1000

# Could initialize to anything 
W1 = np.random.randn(input_size, hidden_size1) * .1
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2) * .1
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size) * .1
b3 = np.zeros((1, output_size))

def relu(Z): 
    return np.maximum(0, Z)

def relu_derivative(Z): 
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def log_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_pred[np.arange(m), Y_true])
    return np.sum(log_likelihood) / m

def binary_label(Y):
    out = np.zeros((Y.size, 10))
    out[np.arange(Y.size), Y] = 1
    return out

def compute_accuracy(predictions, labels):
    return np.mean(predictions == labels)

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backpropagate(X, Y, Z1, A1, Z2, A2, Z3, A3, W2, W3):
    m = Y.shape[0]
    binary_label_Y = binary_label(Y)

    # Output layer
    dZ3 = A3 - binary_label_Y
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # layer 2
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # layer 1
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

# Evaluation
def evaluate_on_test(X_test, Y_test, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_pass(X_test, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(A3, axis=1)
    accuracy = compute_accuracy(predictions, Y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    #wrong_indices = np.where(predictions != Y_test)[0]
    # print(f"Number of incorrect predictions: {len(wrong_indices)}")
    # for i in range(min(5, len(wrong_indices))):
    #     idx = wrong_indices[i]
    #     img = X_test[idx].reshape(28, 28)

    #     plt.imshow(img, cmap='gray')
    #     plt.title(f"Predicted: {predictions[idx]}, Actual: {Y_test[idx]}")
    #     plt.axis('off')
    #     plt.show()

    return accuracy

for epoch in range(epochs):
    Z1, A1, Z2, A2, Z3, A3 = forward_pass(X, W1, b1, W2, b2, W3, b3)

    loss = log_loss(A3, Y)
    predictions = np.argmax(A3, axis=1)
    accuracy = compute_accuracy(predictions, Y)

    dW1, db1, dW2, db2, dW3, db3 = backpropagate(X, Y, Z1, A1, Z2, A2, Z3, A3, W2, W3)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss:.4f} — Accuracy: {accuracy*100:.2f}%")

evaluate_on_test(X_test, Y_test, W1, b1, W2, b2, W3, b3)