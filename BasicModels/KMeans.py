import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" Clustering by Hand - Again - not really the best data to do it with / objective"""


# Load and clean the data
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
df['Diff'] = (df['Strike'] - df['Spot']) / df['Spot']

features = df[['Strike', 'Spot', 'BidAsk', 'Diff(%)']].copy()
features.columns = ['Strike', 'Spot', 'BidAsk', 'Diff']

# Normalize
X = features.values
X = (X - X.mean(axis=0)) / X.std(axis=0)

def kmeans(X, k=3, max_iters=100):
    np.random.seed(42)
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


k = 4 
labels, centroids = kmeans(X, k=k)

df['Cluster'] = labels

cluster_summary = df.groupby('Cluster')[['Strike', 'Spot', 'BidAsk', 'Diff']].mean()
print(cluster_summary)

print("\nC/P by Cluster:")
print(df.groupby(['Cluster', 'C/P']).size())
plt.figure(figsize=(10, 6))
for cluster_id in range(k):
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Spot'], cluster_data['Strike'], label=f"Cluster {cluster_id}", alpha=0.6)

plt.xlabel("Spot Price")
plt.ylabel("Strike Price")
plt.title("Option Trades Clustered (Strike vs Spot)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
