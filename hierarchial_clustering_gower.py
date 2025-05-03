import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import gower

from cluster import get_emotions, get_cohorts
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar
from hierarchial_clustering import plot_dendogram

if __name__ == '__main__':
    # --- Load your data ---
    data = pd.read_csv('./data/HR_data.csv')
    
    # --- Define feature groups (fixed indexing) ---
    continuous_cols = data.columns[:52].tolist()
    string_cat_cols = [data.columns[52], data.columns[53], data.columns[57]]
    numeric_cat_cols = [col for col in data.columns[52:68] if col not in string_cat_cols]
    categorical_cols = string_cat_cols + numeric_cat_cols

    # --- Normalize continuous features ---
    scaler = MinMaxScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])

    # --- Impute continuous and numeric categorical features ---
    imputer = SimpleImputer(strategy='mean')
    data[continuous_cols + numeric_cat_cols] = imputer.fit_transform(data[continuous_cols + numeric_cat_cols])

    for col in string_cat_cols:
        data[col] = data[col].astype('category')

    true_nominal_cols = string_cat_cols

    data[numeric_cat_cols] = scaler.fit_transform(data[numeric_cat_cols])

    cat_feature_flags = [col in true_nominal_cols for col in data.columns]
    distance_matrix = gower.gower_matrix(data, cat_features=cat_feature_flags)

    # --- Perform hierarchical clustering ---
    linkage_matrix = linkage(distance_matrix, method='complete')

    # --- Plot dendrogram ---
    N_leafs = 10
    plot_dendogram(linkage_matrix, N_leafs)

    # --- Flat clustering ---
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

