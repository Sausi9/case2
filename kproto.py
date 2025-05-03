import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from sklearn.feature_selection import mutual_info_classif

def impute_and_scale(df):
    imputer_cont = SimpleImputer(strategy = 'mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cont = df[df.columns[1:52]]
    df_cat = df[df.columns[52:]].apply(lambda col: col.astype('category').cat.codes)
    df_cont_imputed = imputer_cont.fit_transform(df_cont)
    df_cat_imputed = imputer_cat.fit_transform(df_cat)
    X_cont_imputed_scaled = (df_cont_imputed-df_cont_imputed.mean())/df_cont_imputed.std()
    X_imputed = np.concatenate([X_cont_imputed_scaled,df_cat_imputed],axis=1)
    df_imputed = pd.DataFrame(X_imputed, columns=df.columns[1:])
    return df_imputed, X_cont_imputed_scaled

def find_clusters(n_clusters,X):
    kproto = KPrototypes(n_clusters=n_clusters,
                         init='Huang',
                         verbose=0,
                         random_state=21,
                         )
    clusters = kproto.fit_predict(X=X,
                                  categorical=list(range(51,67)))
    return clusters, kproto

def compute_feature_info(df_imputed, labels):
    X = df_imputed.values 
    discrete = [False]*51 + [True]*16
    mi_scores = mutual_info_classif(X, labels, discrete_features=discrete, random_state=0)
    mi_df = pd.DataFrame({
        'feature': df_imputed.columns,
        'mi_score': mi_scores,
        'is_categorical' : discrete
    }).sort_values('mi_score', ascending=False)
    return mi_df

if __name__ == '__main__':
    df = pd.read_csv('data/HR_data.csv')
    df_imputed, df_cont_imputed_scaled = impute_and_scale(df)
    clusters, kproto = find_clusters(n_clusters=2, X = df_imputed)
    score = silhouette_score(df_cont_imputed_scaled, kproto.labels_)
    print(f'Silhouette score is : {score}')
    pca = PCA(n_components = 2)
    pca_data = pca.fit_transform(df_cont_imputed_scaled)
    fig,axs = plt.subplots(1, figsize=(8,6))
    axs.scatter(pca_data[:, 0], pca_data[:, 1], c = kproto.labels_ , cmap = 'viridis', alpha = 0.7)
    plt.show()

    mi_df = compute_feature_info(df_imputed,kproto.labels_)
    print(mi_df)
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_[:2]):.2%}")
