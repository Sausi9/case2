import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import gower

from cluster import get_emotions, get_cohorts, get_phases, get_puzzler
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar, compute_physiological_stats
from hierarchial_clustering import plot_dendogram

if __name__ == '__main__':
    # --- Load your data ---
    data = pd.read_csv('./data/HR_data.csv').iloc[:, 1:]
    data_nonscaled =pd.read_csv('./data/HR_data.csv').iloc[:, 1:]
    # --- Define feature groups (fixed indexing) ---
    continuous_cols = data.columns[:51].tolist()
    string_cat_cols = [data.columns[51], data.columns[52], data.columns[56]]
    numeric_cat_cols = [col for col in data.columns[51:67] if col not in string_cat_cols]
    categorical_cols = string_cat_cols + numeric_cat_cols
    emotion_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                    'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']

    # --- Impute continuous, numeric categorical, and emotion features ---
    imputer_cont = SimpleImputer(strategy='mean')
    imputer_ordinal = SimpleImputer(strategy='median')

    data[continuous_cols] = imputer_cont.fit_transform(data[continuous_cols])
    data[numeric_cat_cols] = imputer_ordinal.fit_transform(data[numeric_cat_cols])
    data[emotion_cols] = imputer_ordinal.fit_transform(data[emotion_cols])

    # --- Clip emotion values to known scales before scaling ---
    data['Frustrated'] = data['Frustrated'].clip(lower=1, upper=10)
    for col in emotion_cols:
        if col != 'Frustrated':
            data[col] = data[col].clip(lower=1, upper=5)

    # --- Normalize continuous and numeric categorical features ---
    scaler = MinMaxScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    # --- Normalize emotion ratings using known fixed scale ---
    scaler_5pt = MinMaxScaler(feature_range=(0, 1))
    scaler_5pt.fit([[1], [5]])
    scaler_10pt = MinMaxScaler(feature_range=(0, 1))
    scaler_10pt.fit([[1], [10]])

    data['Frustrated'] = scaler_10pt.transform(data[['Frustrated']])
    for col in emotion_cols:
        if col != 'Frustrated':
            data[col] = scaler_5pt.transform(data[[col]])

    # --- Ensure string categoricals are categorical dtype ---
    for col in string_cat_cols:
        data[col] = data[col].astype('category')

    # --- PCA on continuous data ---
    continuous_data = data[continuous_cols].to_numpy()
    num_pca_components = 7
    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(continuous_data)
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_[:num_pca_components]):.2%}")

    # --- Scale PCA result ---
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    pca_df = pd.DataFrame(scaler.fit_transform(pca_df), columns=pca_df.columns)

    # --- Combine with categorical data ---
    categorical_df = data[string_cat_cols + numeric_cat_cols].copy()
    for col in string_cat_cols:
        categorical_df[col] = categorical_df[col].astype('category')

    combined_df = pd.concat([pca_df, categorical_df.reset_index(drop=True)], axis=1)
    print(combined_df)

    # --- Compute Gower distance ---
    cat_feature_flags = [False] * pca_df.shape[1] + [True] * categorical_df.shape[1]
    distance_matrix = gower.gower_matrix(combined_df, cat_features=cat_feature_flags)

    # --- Perform hierarchical clustering ---
    linkage_matrix = linkage(distance_matrix, method='ward')

    # --- Plot dendrogram ---
    N_leafs = 10
    plot_dendogram(linkage_matrix, N_leafs)

    # --- Flat clustering ---
    num_clusters = 3
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    data['Cluster'] = cluster_labels

    # --- Emotion statistics ---
    emotions_pd = get_emotions(data_nonscaled)
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
