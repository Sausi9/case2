import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
import gower

from cluster import get_emotions, get_cohorts
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters
from hierarchial_clustering import plot_dendogram

if __name__ == '__main__':
    data = pd.read_csv('./data/HR_data.csv')

    string_cat_cols = [data.columns[52], data.columns[53], data.columns[57]]
    numeric_cat_cols = [col for col in data.columns[52:68] if col not in string_cat_cols]
    categorical_cols = string_cat_cols + numeric_cat_cols

    cat_data = data[categorical_cols].copy()

    for col in string_cat_cols:
        cat_data[col] = cat_data[col].astype('category')

    imputer = SimpleImputer(strategy='most_frequent')
    cat_data[numeric_cat_cols] = imputer.fit_transform(cat_data[numeric_cat_cols])

    cat_feature_flags = [True] * cat_data.shape[1]
    distance_matrix = gower.gower_matrix(cat_data, cat_features=cat_feature_flags)

    linkage_matrix = linkage(distance_matrix, method='complete')

    N_leafs = 10
    plot_dendogram(linkage_matrix, N_leafs)

    num_clusters = 3
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    data['Cluster'] = cluster_labels

    emotions_pd = get_emotions(data)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)

    compute_emotion_statistics(cluster_labels, emotions_pd_imputed)
    count_data_points_per_cluster(cluster_labels)
    cohorts_pd = get_cohorts(data)
    count_unique_per_cluster(cluster_labels, cohorts_pd, 'cohorts')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cluster_labels, y=emotions_pd_imputed['upset'])
    plt.title('Upset Ratings by Cluster (Categorical Features Only)')
    plt.xlabel('Cluster')
    plt.ylabel('Upset Rating')
    plt.tight_layout()
    plt.show()

    compute_silhouette_score(distance_matrix, cluster_labels)
    plot_pca_with_clusters(distance_matrix, cluster_labels)

