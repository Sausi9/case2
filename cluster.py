import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def import_data(pth : str):
    data = pd.read_csv(pth)
    X = data.iloc[:, 1:52].to_numpy()
    return X

def extract_continuous(data):
    continuous_cols = data.columns[1:52]
    continuous_data = data[continuous_cols]
    return continuous_data

def k_means(k, X):
    kmeans = KMeans(n_clusters = k, random_state = 21)
    kmeans.fit(X)
    return kmeans

def get_emotions(data):
    emotions = data.iloc[:, 56:57] 
    emotions = pd.concat([emotions, data.iloc[:, 58:]], axis=1)
    return emotions



if __name__ == "__main__":
    data = pd.read_csv('./data/HR_data.csv')
    # X = import_data('./data/HR_data.csv')
    
    continuous_data = extract_continuous(data)
    imputer = SimpleImputer(strategy = 'mean')
    continuous_data_imputed = imputer.fit_transform(continuous_data)

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(continuous_data_imputed)

    print(continuous_data)

    kmeans = k_means(10, scaled_X)

    data['Cluster'] = kmeans.labels_

    pca = PCA(n_components = 2)
    pca_data = pca.fit_transform(scaled_X)
    pca_centers = pca.transform(kmeans.cluster_centers_)

    score = silhouette_score(scaled_X, kmeans.labels_)  
    print(f"Silhouette Score: {score}")

    plt.figure(figsize = (8,6))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c = data['Cluster'], cmap = 'viridis', alpha = 0.7)
    plt.scatter(pca_centers[:, 0], pca_centers[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.legend(handles=scatter.legend_elements()[0] + [plt.scatter([], [], c='red', marker='x', s=200)], 
            labels=[f'Cluster {i}' for i in range(len(pca_centers))] + ['Centroids'])

    plt.show()

    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_[:2]):.2%}")

