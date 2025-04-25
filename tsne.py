import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def extract_continuous(data):
    continuous_cols = data.columns[1:52]
    continuous_data = data[continuous_cols]
    return continuous_data

def k_means(k, X):
    kmeans = KMeans(n_clusters=k, random_state=21)
    kmeans.fit(X)
    return kmeans

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('./data/HR_data.csv')
    continuous_data = extract_continuous(data)
    imputer = SimpleImputer(strategy='mean')
    continuous_data_imputed = imputer.fit_transform(continuous_data)
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(continuous_data_imputed)

    # Apply KMeans clustering
    k = 3  # Number of clusters
    kmeans = k_means(k, scaled_X)
    data['Cluster'] = kmeans.labels_

    # Prepare data for t-SNE
    n_samples = scaled_X.shape[0]
    combined = np.vstack((scaled_X, kmeans.cluster_centers_))
    tsne = TSNE(n_components=2, random_state=21)
    tsne_combined = tsne.fit_transform(combined)
    tsne_data = tsne_combined[:n_samples, :]
    tsne_centers = tsne_combined[n_samples:, :]

    # Compute silhouette score
    score = silhouette_score(scaled_X, kmeans.labels_)
    print(f"Silhouette Score: {score}")

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
    plt.scatter(tsne_centers[:, 0], tsne_centers[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.legend(handles=scatter.legend_elements()[0] + [plt.scatter([], [], c='red', marker='x', s=200)],
               labels=[f'Cluster {i}' for i in range(k)] + ['Centroids'])
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
