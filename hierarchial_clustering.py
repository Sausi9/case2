import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from cluster import import_data, extract_continuous, get_emotions
import seaborn as sns
from scipy import stats


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
    imputer = SimpleImputer(strategy = 'mean')
    continuous_data_imputed = imputer.fit_transform(continuous_data)

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(continuous_data_imputed)

    d_sample = 'euclidean' 
    d_group = 'ward'
    N_leafs = 10
    Z = linkage(scaled_X, method = d_group, metric = d_sample)
    k = 3
    cluster_designation = fcluster(Z, k, criterion = 'maxclust') 

    #plot_dendogram(Z, N_leafs)

    emotions_pd = get_emotions(data) 
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)
    emotion_scales = {'Frustrated': 10, 'upset': 5, 'hostile': 5, 'alert': 5, 'ashamed': 5, 'inspired': 5, 'nervous': 5, 'attentive': 5, 'afraid': 5, 'active': 5, 'determined': 5}
    emotions = emotions_pd_imputed.to_numpy()

    for i in range(1, k+1):
        datapoints = np.where(cluster_designation == i)[0]  # Get indices directly
        cluster_emotions = emotions[datapoints]
        print(f'Cluster {i} Statistics:')
        for j, emotion in enumerate(emotions_pd_imputed.columns):
            ratings = cluster_emotions[:, j]
            # Check if there are valid ratings left
            if len(ratings) > 0:
                mean_val = np.mean(ratings)
                median_val = np.median(ratings)
                mode_val = pd.Series(ratings).value_counts().index[0]
                std_val = np.std(ratings)
                print(f'{emotion}: Mean = {mean_val:.2f}, Median = {median_val:.2f}, Mode = {mode_val:.2f}, Std = {std_val:.2f}')
            else:
                print(f'{emotion}: No valid data points in this cluster')
        print('\n')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=emotions_pd_imputed, x=cluster_designation, y='Frustrated')
    plt.title('Frustrated Ratings by Cluster')
    plt.show()
