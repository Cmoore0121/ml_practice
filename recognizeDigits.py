import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" Neural Network Classifier on Handwritten Digits 0-9 """
""" 28 x 28 images ; flattened into (Num Examples x 784) data set """
""" Relu + softmax """
""" Simple, 1 hidden layer """


data = pd.read_csv("DigitRecognition/digits28x28.csv").to_numpy()



split_index = int(0.85 * len(data))
train_split = data[:split_index]
test_split = data[split_index:]

# Training set
Y = train_split[:, 0]
X = train_split[:, 1:] / 255.0

# example data point
# num = X[2].reshape(28, 28) * 255.0
# plt.imshow(num, cmap='gray')
# plt.show()

# Couldn't find dataset with just black or white pixels - But we can make it like that if needed
# X = (X > .5).astype(float)

Y_test = test_split[:, 0]
X_test = test_split[:, 1:] / 255.0

#X_test = (X_test > .5).astype(float)


input_size = 784
hidden_size = 64
output_size = 10
learning_rate = 0.1
epochs = 1000

W1 = np.random.randn(input_size, hidden_size) * 0.01 
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def log_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_pred[range(m), Y_true])
    return np.sum(log_likelihood) / m

def binary_label(Y):
    binary_label_Y = np.zeros((Y.size, 10))
    binary_label_Y[np.arange(Y.size), Y] = 1
    return binary_label_Y

def forward_pass(X, W1, B1, W2, B2):
    layer_1 = X @ W1 + B1
    activation_1 = relu(layer_1)
    layer_2 = activation_1 @ W2 + B2
    activation_final = softmax(layer_2)
    return layer_1, activation_1, layer_2, activation_final

def backpropagate(X, Y, Z1, A1, Z2, A2, W2):
    num_examples = Y.shape[0]
    binary_label_Y = binary_label(Y)

    dL_dZ2 = A2 - binary_label_Y                  # (N x 10)
    dL_dW2 = (A1.T @ dL_dZ2) / num_examples      # (64 x N) @ (N x 10) => (64 x 10)
    db2 = np.sum(dL_dW2, axis=0, keepdims=True) / num_examples  # (1 x 10)

    # dL_dZ1 = dL_dA1 * dA1_dZ1 etc. 
    # Chain rule math in notes

    dL_dA1 = dL_dZ2 @ W2.T                        # (N x 10) @ (10 x 64) => (N x 64)
    dL_dZ1 = dL_dA1 * relu_derivative(Z1)
    dL_dW1 = X.T @ dL_dZ1 / num_examples                     # (784 x N) @ (N x 64) => (784 x 64)
    db1 = np.sum(dL_dZ1, axis=0, keepdims=True) / num_examples  # (1 x 64)

    return dL_dW1, db1, dL_dW2, db2


def compute_accuracy(predictions, labels):
    return np.mean(predictions == labels)


def evaluate_on_test(X_test, Y_test, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X_test, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    accuracy = compute_accuracy(predictions, Y_test)
    wrong_indices = np.where(predictions != Y_test)[0]
    print(f"Number of incorrect predictions: {len(wrong_indices)}")
    for i in range(min(5, len(wrong_indices))):
        idx = wrong_indices[i]
        img = X_test[idx].reshape(28, 28)

        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {predictions[idx]}, Actual: {Y_test[idx]}")
        plt.axis('off')
        plt.show()

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    return accuracy, wrong_indices


for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)
    loss = log_loss(A2, Y)
    predictions = np.argmax(A2, axis=1)
    accuracy = compute_accuracy(predictions, Y)
    dW1, db1, dW2, db2 = backpropagate(X, Y, Z1, A1, Z2, A2, W2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss:.4f} — Accuracy: {accuracy*100:.2f}%")

evaluate_on_test(X_test, Y_test, W1, b1, W2, b2)



""" If wanted to do it using one example at a time """
# for epoch in range(epochs):
#     dW1 = np.zeros_like(W1)
#     db1 = np.zeros_like(b1)
#     dW2 = np.zeros_like(W2)
#     db2 = np.zeros_like(b2)
#     total_loss = 0
#     correct = 0

#     for i in range(len(X)):
#         x_i = X[i].reshape(1, -1)  # shape (1, 784)
#         y_i = Y[i]

#         # Forward 
#         z1 = x_i @ W1 + b1       # (1 x 64)
#         a1 = relu(z1)            # (1 x 64)
#         z2 = a1 @ W2 + b2        # (1 x 10)
#         a2 = softmax(z2)         # (1 x 10)

#         pred_label = np.argmax(a2)
#         if pred_label == y_i:
#             correct += 1

#         total_loss += -np.log(a2[0, y_i])

#         binary_label = np.zeros((1, 10))
#         binary_label[0, y_i] = 1

#         # Backward

#         dz2 = a2 - binary_label                 # (1 x 10)
#         dW2 += a1.T @ dz2                   # (64 x 1) @ (1 x 10)
#         db2 += dz2                          # (1 x 10)

#         da1 = dz2 @ W2.T                    # (1 x 64)
#         dz1 = da1 * relu_derivative(z1)     # (1 x 64)
#         dW1 += x_i.T @ dz1                  # (784 x 1) @ (1 x 64)
#         db1 += dz1                          # (1 x 64)

#  
#     m = len(X)
#     W1 -= learning_rate * dW1 / m
#     b1 -= learning_rate * db1 / m
#     W2 -= learning_rate * dW2 / m
#     b2 -= learning_rate * db2 / m


#     avg_loss = total_loss / m
#     accuracy = correct / m
#     print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Accuracy: {accuracy*100:.2f}%")



