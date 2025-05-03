import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from cluster import import_data, extract_continuous, get_emotions, get_puzzler, get_cohorts
import seaborn as sns
from scipy import stats
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters


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


if __name__ == "__main__":
    data = pd.read_csv('./data/HR_data.csv')
    continuous_data = extract_continuous(data)
    imputer = SimpleImputer(strategy='mean')
    continuous_data_imputed = imputer.fit_transform(continuous_data)

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(continuous_data_imputed)

    d_sample = 'euclidean'
    d_group = 'ward'
    N_leafs = 10
    Z = linkage(scaled_X, method=d_group, metric=d_sample)
    k = 2
    cluster_designation = fcluster(Z, k, criterion='maxclust')

    # Uncomment to plot dendrogram if needed
    # plot_dendogram(Z, N_leafs)

    # Prepare emotions data
    emotions_pd = get_emotions(data)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)
    emotion_scales = {'Frustrated': 10, 'upset': 5, 'hostile': 5, 'alert': 5, 'ashamed': 5, 'inspired': 5, 'nervous': 5, 'attentive': 5, 'afraid': 5, 'active': 5, 'determined': 5}

    # Compute emotion statistics
    compute_emotion_statistics(cluster_designation, emotions_pd_imputed)

    # Count data points per cluster (replacing puzzlers count)
    count_data_points_per_cluster(cluster_designation)

    # Count unique cohorts per cluster
    cohorts_pd = get_cohorts(data)
    count_unique_per_cluster(cluster_designation, cohorts_pd, 'cohorts')

    # Boxplot for 'upset' ratings by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=emotions_pd_imputed, x=cluster_designation, y='upset')
    plt.title('Upset Ratings by Cluster')  # Corrected title from 'Frustrated' to 'Upset'
    plt.show()

    # Compute silhouette score
    compute_silhouette_score(scaled_X, cluster_designation)

    # Plot PCA with clusters
    plot_pca_with_clusters(scaled_X, cluster_designation)
