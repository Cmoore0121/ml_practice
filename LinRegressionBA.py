import numpy as np
import pandas as pd


""" Given various data points - predict the bid/Ask price"""
""" Linear Regression using MSE and also using Moore Penrose Pseudoinverse """
""" Not really great data, but more to do a linear regression by hand (not the PInverse Part)"""


df = pd.read_csv("BasicModels/optionTradeData.csv")
def parse_number(x):
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('K'):
            return float(x[:-1]) * 1e3
        elif x.endswith('M'):
            return float(x[:-1]) * 1e6
        try:
            return float(x)
        except ValueError:
            return np.nan
    return x

df['Vol'] = df['Vol'].apply(parse_number)
df['Prems'] = df['Prems'].apply(parse_number)
df['OI'] = df['OI'].apply(parse_number)
df = df.dropna()

df['Time'] = pd.to_datetime(df['Time'])
df['Exp'] = pd.to_datetime(df['Exp'])
df['Dte'] = (df['Exp'] - df['Time']).dt.days / 365
df['Dte'] = df['Dte'].map(lambda dt: 0 if dt < 0 else dt)
df['Diff'] = (df['Strike'] - df['Spot']) / df['Spot']

df_encoded = pd.get_dummies(df[['C/P']], drop_first=True)

X = pd.concat([
    df[['Diff', 'Dte', 'Orders', 'Vol', 'OI']],
    df_encoded
], axis=1).values
y = df['BidAsk'].values.reshape(-1, 1)

X = X.astype(float)
y = y.astype(float)
X = (X - X.mean(axis=0)) / X.std(axis=0)

X = np.hstack([np.ones((X.shape[0], 1)), X])

split_idx = int(0.8 * X.shape[0])
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

#normalize
mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0)
std[std == 0] = 1
X_train = (X_train_raw - mean) / std
X_test = (X_test_raw - mean) / std

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

def compute_mse(X, y, weights):
    m = len(y)
    preds = X @ weights
    return np.sum((preds - y) ** 2) / m

# Gradient Descent for Linear Regression
def gradient_descent_linear(X, y, learning_rate=0.01, iters=2000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    for i in range(iters):
        preds = X @ weights
        gradient = (2/m) * X.T @ (preds - y)
        weights -= learning_rate * gradient
        if i % 200 == 0:
            print("MSE:", compute_mse(X, y, weights))
    return weights

weights = gradient_descent_linear(X_train, y_train)
y_pred = X_test @ weights
mse_test = compute_mse(X_test, y_test, weights)
print("Test MSE:", mse_test)


import matplotlib.pyplot as plt


actual = y_test[:50].flatten()
predicted = y_pred[:50].flatten()

plt.figure(figsize=(10, 6))
plt.plot(range(50), actual, label="Actual", marker='o')
plt.plot(range(50), predicted, label="Predicted", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Option Premium")
plt.title("First 50 Predictions: Actual vs Predicted Option Bid/Ask")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# lin alg using pseudo inverse
weights_pinv = np.linalg.pinv(X_train) @ y_train
y_pred_pinv = X_test @ weights_pinv


mse_pinv = compute_mse(X_test, y_test, weights_pinv)
print("Test MSE (Pseudoinverse):", mse_pinv)
print(weights_pinv)

