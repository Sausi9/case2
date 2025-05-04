import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
import seaborn as sns

from cluster import get_emotions, get_cohorts, get_phases
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar
from preprocessing import load_and_preprocess_data

def find_clusters(n_clusters, X):
    kproto = KPrototypes(n_clusters=n_clusters,
                         init='Huang',
                         verbose=0,
                         random_state=21)
    clusters = kproto.fit_predict(X=X, categorical=list(range(X.shape[1] - 16, X.shape[1])))
    return clusters, kproto

def compute_feature_info(df_imputed, labels):
    X = df_imputed.values 
    discrete = [False]* (df_imputed.shape[1] - 16) + [True]*16
    mi_scores = mutual_info_classif(X, labels, discrete_features=discrete, random_state=0)
    mi_df = pd.DataFrame({
        'feature': df_imputed.columns,
        'mi_score': mi_scores,
        'is_categorical': discrete
    }).sort_values('mi_score', ascending=False)
    return mi_df

if __name__ == '__main__':
    # Load and preprocess the data using the pipeline
    raw_data, combined_df, nominal_cat_cols, ordinal_cat_cols, binary_cat_cols, emotion_cols, cat_feature_flags = load_and_preprocess_data('data/HR_data.csv', use_pca=False)
    # Run K-Prototypes clustering
    clusters, kproto = find_clusters(n_clusters=2, X=combined_df.to_numpy())
    # Compute silhouette score on the continuous features only (first 51 columns)
    cont_scaled = combined_df.iloc[:, :51].to_numpy()

    # Mutual information feature importance
    mi_df = compute_feature_info(combined_df, kproto.labels_)
    print(mi_df)

    # Emotion statistics and plots
    emotions_pd = get_emotions(raw_data)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)

    compute_emotion_statistics(kproto.labels_, emotions_pd_imputed)
    count_data_points_per_cluster(kproto.labels_)
    cohorts_pd = get_cohorts(raw_data)
    count_unique_per_cluster(kproto.labels_, cohorts_pd, 'cohorts')

    phases_pd = get_phases(raw_data)
    count_unique_per_cluster(kproto.labels_, phases_pd, 'phase')

    # Boxplot for 'upset' by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=kproto.labels_, y=emotions_pd_imputed['upset'])
    plt.title('Upset Ratings by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Upset Rating')
    plt.tight_layout()
    plt.show()

    # Optional: Silhouette score & PCA visualization
    compute_silhouette_score(cont_scaled, kproto.labels_)
    plot_pca_with_clusters(cont_scaled, kproto.labels_)
    plot_emotion_radar(emotions_pd_imputed, kproto.labels_)

