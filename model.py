import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def segment_customers(data_path, n_clusters=3):
    df = pd.read_csv(data_path)
    features = df[['annual_income', 'spending_score', 'age']]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    print(f"Customers successfully segmented into {n_clusters} groups.")
    return df

if __name__ == "__main__":
    print("Running Customer Segmentation Engine...")
