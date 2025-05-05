import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from kmodes.util.dissim import jaccard_dissim_label
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
import seaborn as sns

from cluster import get_emotions, get_cohorts, get_phases
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters, plot_emotion_radar
from preprocessing import load_and_preprocess_data

def kproto_preprocess(df):
    cont_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cont_cols = df.columns[:51]
    cat_cols = df.columns[52:]
    df[cont_cols] = cont_imputer.fit_transform(df[cont_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    ord_cols = ['upset', 'hostile', 'alert', 'ashamed',
                'inspired', 'nervous', 'attentive', 'afraid',
                'active', 'determined']
    df['Frustrated'] = (df['Frustrated']-0.5)/11.0
    df[ord_cols] = df[ord_cols].apply(lambda x: (x-0.5)/6.0)
    cat_cols_num = [51, 52, 53, 54, 56]
    cat_cols_reorder = [55] + list(range(57,df.shape[1])) + cat_cols_num
    all_cols = df.columns.tolist()
    front = all_cols[:51]
    tail = [ all_cols[i] for i in cat_cols_reorder ]
    new_order = front + tail
    df = df[new_order]
    new_cat_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
    df[new_cat_cols]  = df[new_cat_cols].apply(lambda col: col.astype('category').cat.codes)
    cont_cols_new = df.columns[:df.shape[1]-5]
    df[cont_cols_new] = (df[cont_cols_new] - df[cont_cols_new].mean())/df[cont_cols_new].std()
    return df 

def plot_individuals_bar(df,labels):
    df['cluster'] = labels + 1 
    counts = (
        df
        .groupby(['Individual', 'cluster'])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.sort_index()
    fig, ax = plt.subplots(figsize=(12,6))
    counts.plot(kind='bar', stacked=True, ax=ax)

    ax.set_xlabel('Individual ID')
    ax.set_ylabel('Frequency')
    ax.set_title('How Often Each Individual Appears in Each Cluster')
    ax.legend(title='Cluster')

    plt.tight_layout()
    plt.show()

def find_clusters(n_clusters, X):
    kproto = KPrototypes(n_clusters=n_clusters,
                         init='Cao',
                         verbose=0,
                         random_state=21,
                         max_iter = 1000,
                         n_init = 20,
                         cat_dissim=jaccard_dissim_label)
    clusters = kproto.fit_predict(X=X, categorical=list(range(X.shape[1] - 5, X.shape[1])))
    return clusters, kproto

def compute_feature_info(df_imputed, labels):
    X = df_imputed.values 
    discrete = [False]* (df_imputed.shape[1] - 5) + [True]*5
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
    df_kproto = kproto_preprocess(raw_data)
    # Run K-Prototypes
    clusters, kproto = find_clusters(n_clusters=3, X=df_kproto)
    # Compute silhouette score on the continuous features only
    cont_scaled = df_kproto.iloc[:, :df_kproto.shape[1] - 5].to_numpy()
    print(df_kproto.iloc[:,:df_kproto.shape[1]-5].mean())
    # Mutual information feature importance
    mi_df = compute_feature_info(df_kproto, kproto.labels_)
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
    plot_individuals_bar(df_kproto,kproto.labels_)
