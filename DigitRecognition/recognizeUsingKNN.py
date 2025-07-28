import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

""" same objective - just using KNN """


data = pd.read_csv("DigitRecognition/digits28x28.csv").to_numpy()
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

X = train_data[:, 1:] / 255.0
Y = train_data[:, 0]
X_test = test_data[:, 1:] / 255.0
Y_test = test_data[:, 0]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(x, X_train, Y_train, k=3):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(x, X_train[i])
        distances.append((dist, Y_train[i]))

    # Sort by distance 
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

def knn_test(X_train, Y_train, X_test, Y_test, k=3):
    correct = 0
    total = len(X_test)

    for i in range(total):
        pred = knn_predict(X_test[i], X_train, Y_train, k)
        if pred == Y_test[i]:
            correct += 1
        if i % 100 == 0:
            print(f"Tested {i}/{total} samples...")

    accuracy = correct / total
    print(f"\nFinal Accuracy with k={k}: {accuracy * 100:.2f}%")
    return accuracy

knn_test(X, Y, X_test[:200], Y_test[:200], k=3)
knn_test(X, Y, X_test[:200], Y_test[:200], k=5)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, Y)
accuracy = rf.score(X_test, Y_test)
print(accuracy)