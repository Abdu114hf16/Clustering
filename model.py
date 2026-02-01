import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

mall_data = pd.read_csv('Mall_Customers.csv')

income_spending = mall_data.iloc[:, [2, 3]].values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(income_spending)

kmeans_engine = KMeans(n_clusters=5, init='k-means++', random_state=42)
cluster_labels = kmeans_engine.fit_predict(scaled_features)

score = silhouette_score(scaled_features, cluster_labels)

mall_data['ClusterGroup'] = cluster_labels
print(mall_data.head())
print(f"Clustering Performance (Silhouette Score): {score:.4f}")
