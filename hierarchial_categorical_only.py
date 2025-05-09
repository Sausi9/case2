import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
import gower

from cluster import get_emotions, get_cohorts, get_phases, get_puzzler
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar, compute_physiological_stats
from hierarchial_final import plot_dendogram

if __name__ == '__main__':
    data = pd.read_csv('./data/HR_data.csv').iloc[:, 1:]

    string_cat_cols = [data.columns[51], data.columns[52], data.columns[56]]
    numeric_cat_cols = [col for col in data.columns[51:67] if col not in string_cat_cols]
    categorical_cols = string_cat_cols + numeric_cat_cols

    cat_data = data[categorical_cols].copy()

    for col in string_cat_cols:
        cat_data[col] = cat_data[col].astype('category')

    imputer = SimpleImputer(strategy='most_frequent')
    cat_data[numeric_cat_cols] = imputer.fit_transform(cat_data[numeric_cat_cols])

    cat_feature_flags = [True] * cat_data.shape[1]
    distance_matrix = gower.gower_matrix(cat_data, cat_features=cat_feature_flags)

    linkage_matrix = linkage(distance_matrix, method='ward')

    N_leafs = 10
    plot_dendogram(linkage_matrix, N_leafs)

    num_clusters = 3
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    data['Cluster'] = cluster_labels

    # --- Emotion statistics ---
    emotions_pd = get_emotions(data)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)

    # Compute statistics and plots
    compute_emotion_statistics(cluster_labels, emotions_pd_imputed)
    count_data_points_per_cluster(cluster_labels)
    cohorts_pd = get_cohorts(data)
    count_unique_per_cluster(cluster_labels, cohorts_pd, 'cohorts')

    phases_pd = get_phases(data)
    count_unique_per_cluster(cluster_labels, phases_pd, 'phase')

    puzzler_pd = get_puzzler(data)
    count_unique_per_cluster(cluster_labels, puzzler_pd, 'puzzler')
    compute_physiological_stats(data, cluster_labels)

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

