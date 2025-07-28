import numpy as np
import pandas as pd

""" Given various data points about a trade and the result of if the option expired ITM or not later in time"""
""" Logistic Regression to predict ITM given current scenario"""
""" Not really great data, but more to do a logistic regression by hand"""


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
df['Dte'] = (df['Exp'] - df['Time']).dt.days/365
df['Dte'] = df['Dte'].map(lambda dt: 0 if dt < 0 else dt)
df['Diff'] = (df['Strike'] - df['Spot'])/df['Spot']
df_encoded = pd.get_dummies(df[['C/P']], drop_first=True)

X = pd.concat([
    df[['Diff', 'Dte', 'BidAsk', 'Orders', 'Vol', 'Prems', 'OI']],
    df_encoded
], axis=1).values
y = df['ITM'].values.reshape(-1, 1)

X = X.astype(float)
X = (X - X.mean(axis=0)) / X.std(axis=0)

X = np.hstack([np.ones((X.shape[0], 1)), X])

split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    epsilon = 1e-5
    cost = -(1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    return cost.item()

def gradient_descent(X, y, learnign_rate=0.1, iters=2000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    for _ in range(iters):
        h = sigmoid(X @ weights)
        gradient = X.T @ (h - y) / m
        weights -= learnign_rate * gradient
    return weights

def predict(X, weights, threshold=0.5):
    return (sigmoid(X @ weights) >= threshold).astype(int)

weights = gradient_descent(X_train, y_train)
y_pred = predict(X_test, weights)

accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)



# To see how well it compares to RF Classification

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train.ravel())
y_pred = rf_model.predict(X_test_scaled)

report1 = classification_report(y_test, y_pred, output_dict = True)
print("Accuracy:", report1["accuracy"])
pd.DataFrame(report1)