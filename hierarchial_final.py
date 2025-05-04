import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import gower

from cluster import get_emotions, get_cohorts, get_phases, get_puzzler
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar, compute_physiological_stats
from preprocessing import load_and_preprocess_data

def plot_dendogram(Z, N_leafs):
    plt.figure()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    den = dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        truncate_mode='lastp',
        p = N_leafs,
    )
    plt.show()

if __name__ == '__main__':
    # --- Load and preprocess data ---
    raw_data, combined_df, nominal_cat_cols, ordinal_cat_cols, binary_cols, emotion_cols, cat_feature_flags = load_and_preprocess_data('./data/HR_data.csv', n_pca_components=10, use_pca = True)
    # --- Compute Gower distance ---
    distance_matrix = gower.gower_matrix(combined_df, cat_features=cat_feature_flags)

    # --- Perform hierarchical clustering ---
    linkage_matrix = linkage(distance_matrix, method='ward')

    # --- Plot dendrogram ---
    N_leafs = 15
    plot_dendogram(linkage_matrix, N_leafs)

    # --- Flat clustering ---
    num_clusters = 3
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    raw_data['Cluster'] = cluster_labels

    # --- Emotion statistics ---
    emotions_pd = get_emotions(raw_data)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)

    # Compute statistics and plots
    compute_emotion_statistics(cluster_labels, emotions_pd_imputed)
    count_data_points_per_cluster(cluster_labels)
    cohorts_pd = get_cohorts(raw_data)
    count_unique_per_cluster(cluster_labels, cohorts_pd, 'cohorts')

    phases_pd = get_phases(raw_data)
    count_unique_per_cluster(cluster_labels, phases_pd, 'phase')

    puzzler_pd = get_puzzler(raw_data)
    count_unique_per_cluster(cluster_labels, puzzler_pd, 'puzzler')
    compute_physiological_stats(raw_data, cluster_labels)

    # Boxplot for 'upset' by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cluster_labels, y=emotions_pd_imputed['upset'])
    plt.title('Upset Ratings by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Upset Rating')
    plt.tight_layout()
    plt.show()

    # Optional: Silhouette score & PCA visualization
    compute_silhouette_score(distance_matrix, cluster_labels)
    plot_pca_with_clusters(distance_matrix, cluster_labels)
    plot_emotion_radar(emotions_pd_imputed, cluster_labels)
