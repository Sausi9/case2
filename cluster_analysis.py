import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def compute_emotion_statistics(cluster_designation, emotions_pd_imputed):
    """Compute and print statistics (mean, median, mode, std) for each emotion in each cluster."""
    unique_clusters = np.unique(cluster_designation)
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_designation == cluster)[0]
        cluster_emotions = emotions_pd_imputed.iloc[cluster_idx]
        print(f'Cluster {cluster} Statistics:')
        for emotion in emotions_pd_imputed.columns:
            ratings = cluster_emotions[emotion]
            if len(ratings) > 0:
                mean_val = ratings.mean()
                median_val = ratings.median()
                mode_val = ratings.mode().iloc[0] if not ratings.mode().empty else np.nan
                std_val = ratings.std()
                print(f'{emotion}: Mean = {mean_val:.2f}, Median = {median_val:.2f}, Mode = {mode_val:.2f}, Std = {std_val:.2f}')
            else:
                print(f'{emotion}: No valid data points in this cluster')
        print('\n')

def count_data_points_per_cluster(cluster_designation):
    """Count and print the number of data points in each cluster."""
    unique_clusters = np.unique(cluster_designation)
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_designation == cluster)[0]
        print(f'Cluster {cluster} data points count:')
        print(len(cluster_idx))

def count_unique_per_cluster(cluster_designation, items_series, item_name):
    """Count and print the unique values and their frequencies for a categorical variable in each cluster."""
    unique_clusters = np.unique(cluster_designation)
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_designation == cluster)[0]
        cluster_items = items_series.iloc[cluster_idx]
        print(f'Cluster {cluster} unique {item_name}:')
        print(cluster_items.value_counts())

def compute_silhouette_score(scaled_X, cluster_designation):
    """Compute and print the silhouette score for the clustering."""
    score = silhouette_score(scaled_X, cluster_designation, metric='manhattan')
    print(f"Silhouette Score: {score:.3f}")
    return score

def plot_pca_with_clusters(scaled_X, cluster_designation):
    """Generate and display a PCA plot with data points and cluster centers."""
    unique_clusters = np.unique(cluster_designation)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_X)
    
    cluster_centers = []
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_designation == cluster)[0]
        if len(cluster_idx) > 0:
            center = np.mean(scaled_X[cluster_idx], axis=0)
            cluster_centers.append(center)
        else:
            print(f'Cluster {cluster} is empty')
    cluster_centers = np.array(cluster_centers)
    
    pca_centers = pca.transform(cluster_centers)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_designation, cmap='viridis', alpha=0.6, label='Data Points')
    plt.scatter(pca_centers[:, 0], pca_centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
    plt.title('PCA Projection of Clusters with Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.show()
    
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_[:2]):.2%}")


def plot_emotion_radar(emotions_df, cluster_labels):
    emotions_df = emotions_df.copy()
    emotions_df['Cluster'] = cluster_labels
    cluster_means = emotions_df.groupby('Cluster').mean()

    categories = cluster_means.columns.tolist()
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, row in cluster_means.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title('Emotion Profiles by Cluster')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
