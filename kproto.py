import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from cluster import get_emotions, get_cohorts, get_phases
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('data/HR_data.csv')
    imputer_cont = SimpleImputer(strategy = 'mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cont = df[df.columns[1:52]]
    df_cat = df[df.columns[52:]]
    df_cont_imputed = imputer_cont.fit_transform(df_cont)
    df_cat_imputed = imputer_cat.fit_transform(df_cat)
    df_cont_imputed_scaled = (df_cont_imputed-df_cont_imputed.mean())/df_cont_imputed.std()
    X_imputed = np.concatenate([df_cont_imputed_scaled,df_cat_imputed],axis=1)
    df_imputed = pd.DataFrame(X_imputed, columns=df.columns[1:]) 
    kproto = KPrototypes(n_clusters=3,
                         init='Cao',
                         verbose=0,
                         random_state=21,
                         )
    clusters = kproto.fit_predict(X=df_imputed,
                                  categorical=list(range(51,67)))

    score = silhouette_score(df_cont_imputed_scaled, kproto.labels_) 
    print(f'Silhouette score: {score}')
    pca = PCA(n_components = 2)
    pca_data = pca.fit_transform(df_cont_imputed_scaled)

    plt.figure(figsize = (8,6))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c = kproto.labels_ , cmap = 'viridis', alpha = 0.7)

    plt.legend(handles=scatter.legend_elements()[0] + [plt.scatter([], [], c='red', marker='x', s=200)])

    plt.show()

    # --- Emotion statistics ---
    emotions_pd = get_emotions(df)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)

    # Compute statistics and plots
    compute_emotion_statistics(kproto.labels_, emotions_pd_imputed)
    count_data_points_per_cluster(kproto.labels_)
    cohorts_pd = get_cohorts(df)
    count_unique_per_cluster(kproto.labels_, cohorts_pd, 'cohorts')


    phases_pd = get_phases(df)
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
    compute_silhouette_score(df_cont_imputed_scaled, kproto.labels_)
    plot_pca_with_clusters(df_cont_imputed_scaled, kproto.labels_)

    plot_emotion_radar(emotions_pd_imputed, kproto.labels_)

    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_[:2]):.2%}")
