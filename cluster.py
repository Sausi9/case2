import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from cluster_analysis import compute_emotion_statistics, count_data_points_per_cluster, count_unique_per_cluster, compute_silhouette_score, plot_pca_with_clusters


def import_data(pth : str):
    data = pd.read_csv(pth)
    X = data.iloc[:, 1:52].to_numpy()
    return X

def extract_continuous(data):
    continuous_cols = data.columns[1:52]
    continuous_data = data[continuous_cols]
    return continuous_data

def k_means(k, X):
    kmeans = KMeans(n_clusters = k, random_state = 21)
    kmeans.fit(X)
    return kmeans

def get_emotions(data):
    emotions = data.iloc[:, 55:56] 
    emotions = pd.concat([emotions, data.iloc[:, 57:]], axis=1)
    return emotions

def get_puzzler(data):
    puzzler = data.iloc[:, 54] 
    return puzzler

def get_cohorts(data):
    cohorts = data.iloc[:, 56]
    return cohorts

def get_phases(data):
    phases = data.iloc[:, 52]
    return phases

def select_k(X, k_values, Nsim = 20):
    minX = list(np.min(X, axis=0)) # data range min
    maxX = list(np.max(X, axis=0)) # data range max

    W = np.zeros(len(k_values))
    Wu = np.zeros((len(k_values), Nsim))
    
    for k in k_values:
        kmeans = k_means(k,X) 
        y_pred = kmeans.predict(X)
        for i in range(k):
            idx = np.where(y_pred == i)[0]
            W[i] = (1/2*len(idx))*np.sum(np.linalg.norm(X[idx] - kmeans.cluster_centers_[i], axis=1)**2)
            for j in range(Nsim):
                Xu = np.ones((X.shape[0], 1)) * minX + np.random.rand(X.shape[0], X.shape[1]) * (np.ones((X.shape[0], 1)) * maxX - np.ones((X.shape[0], 1)) * minX)
                kmeans_sim = k_means(k, Xu)
                y_pred_sim = kmeans_sim.predict(Xu)

                for i_sim in range(k):
                    idx_sim = np.where(y_pred_sim == i_sim)[0]
                    Wu[k - 1, j] += (1 / (2 * len(idx_sim))) * np.sum(
                    np.linalg.norm(Xu[idx_sim] - kmeans_sim.cluster_centers_[i_sim], axis = 1) ** 2
                )

        gap_statistic = np.log(Wu[k - 1, :]) - np.log(W[k - 1])

    Elog_Wu = np.mean(np.log(Wu), axis = 1)
    sk = np.std(np.log(Wu), axis=1)*np.sqrt(1+1/Nsim) 
    x_range = np.array(range(len(k_values))) + 1
    plt.figure()
    plt.title("Within-class dissimilarity")
    plt.plot(x_range, np.log(W), label='observed')
    plt.plot(x_range, Elog_Wu, label='expected for simulation')
    plt.legend(loc='upper left')
    plt.xlabel("Number of clusters - k")
    plt.ylabel("log(W)")
    plt.show()
    plt.figure()
    plt.title('Gap curve')
    Gk =  Elog_Wu.T - np.log(W)
    plt.plot(x_range,Gk,color='green')
    x_range_list = []
    x_range_list.append(x_range)
    x_range_list.append(x_range)
    GkList = []
    GkList.append(Gk-sk)
    GkList.append(Gk+sk)
    plt.plot(x_range_list, GkList, color='orange')
    plt.ylabel('G(k)+/- s_k')
    plt.xlabel('number of clusters - k')
    plt.show()
    K_opt = np.where(np.array(Gk[:-1]) >= np.array(Gk[1:] - sk[1:]))[0]

    if not K_opt.size:
        K_opt = clustersNr
        print ("Gap-statistic, optimal K = %d" % K_opt)
    else:    
        print ("Gap-statistic, optimal K = %d" % k_values[K_opt[0]])
    K_opt = k_values[K_opt[0]]
    return K_opt


if __name__ == "__main__":
    data = pd.read_csv('./data/HR_data.csv')
    data2 = pd.read_csv('./data/HR_data.csv').iloc[:, 1:]

    continuous_data = extract_continuous(data)
    imputer = SimpleImputer(strategy = 'mean')
    continuous_data_imputed = imputer.fit_transform(continuous_data)

    #scaler = StandardScaler()
    #scaled_X = scaler.fit_transform(continuous_data_imputed)

    scaled_X = (continuous_data_imputed - continuous_data_imputed.mean())/continuous_data_imputed.std()

    k_range =  range(1, 11)

    k_opt = select_k(scaled_X, k_range)
    kmeans = k_means(k_opt, scaled_X)

    emotions_pd = get_emotions(data2)
    print(emotions_pd)
    imputer = SimpleImputer(strategy='mean')
    emotions_pd_imputed = pd.DataFrame(imputer.fit_transform(emotions_pd), columns=emotions_pd.columns)
    emotion_scales = {'Frustrated': 10, 'upset': 5, 'hostile': 5, 'alert': 5, 'ashamed': 5, 'inspired': 5, 'nervous': 5, 'attentive': 5, 'afraid': 5, 'active': 5, 'determined': 5}

    # Compute emotion statistics
    compute_emotion_statistics(kmeans.labels_, emotions_pd_imputed)

    # Count data points per cluster (replacing puzzlers count)
    count_data_points_per_cluster(kmeans.labels_)

    # Count unique cohorts per cluster
    cohorts_pd = get_cohorts(data)
    count_unique_per_cluster(kmeans.labels_, cohorts_pd, 'cohorts')

    phases_pd = get_phases(data)
    count_unique_per_cluster(kmeans.labels_, phases_pd, 'phase')

    # Boxplot for 'upset' ratings by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=emotions_pd_imputed, x=kmeans.labels_, y='upset')
    plt.title('Upset Ratings by Cluster')  # Corrected title from 'Frustrated' to 'Upset'
    plt.show()

    # Compute silhouette score
    compute_silhouette_score(scaled_X, kmeans.labels_)

    # Plot PCA with clusters
    plot_pca_with_clusters(scaled_X, kmeans.labels_)

