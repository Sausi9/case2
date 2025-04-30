from typing import Tuple, List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans    # swap for KMeans, HDBSCAN, etc.
from sklearn.metrics import silhouette_score


RANDOM_STATE = 42   # keeps results reproducible


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Heuristically separate ordinary numeric features from emotion-rating
    columns.  Tweak the heuristic or pass explicit lists if you prefer.
    """
    # ↳ emotion terms in lower case for case-insensitive match
    emotion_words = {
        "frustrated", "upset", "hostile", "alert", "ashamed",
        "inspired", "nervous", "attentive", "afraid", "active", "determined",
    }

    num_cols  = []
    emo_cols  = []

    for c in df.select_dtypes("number").columns:
        (emo_cols if c.lower() in emotion_words else num_cols).append(c)

    return num_cols, emo_cols


def build_pipeline(df: pd.DataFrame,
                   n_clusters: int = 4,
                   pca_var: float = 0.5) -> Pipeline:
    """
    Assemble an impute → scale → (optional PCA) → cluster pipeline.

    Parameters
    ----------
    df          : pandas DataFrame containing the raw data.
    n_clusters  : number of clusters for AgglomerativeClustering.
    pca_var     : fraction of total variance to preserve (0–1).  Set
                  to None to skip PCA.

    Returns
    -------
    pipe        : an unfitted sklearn Pipeline object.
    """
    num_cols, emo_cols = split_columns(df)

    # --- block-specific preprocessing ---------------------------------------
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler()),
    ])

    emo_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler()),
    ])

    pre = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("emo", emo_pipe, emo_cols),
        ],
        remainder="drop",
    )

    # --- optional dimensionality reduction ---------------------------------
    steps = [("pre", pre)]
    if pca_var:
        steps.append(("pca", PCA(n_components=pca_var,
                                 svd_solver="full",
                                 random_state=RANDOM_STATE)))

    # --- clustering stage ---------------------------------------------------
    steps.append((
    "cluster",
    KMeans(
        n_clusters=n_clusters,          # same parameter you pass to build_pipeline
        n_init=20,                      # robust centroid initialisation
        random_state=RANDOM_STATE,      # reproducible results
    )
    ))

    return Pipeline(steps)

from sklearn.metrics import silhouette_score

def compute_silhouette(pipeline, raw_df) -> float:
    """
    Return the mean silhouette coefficient of the clustering result.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        A *fitted* pipeline whose final step produces `labels_`
        (e.g. KMeans, AgglomerativeClustering, GaussianMixture).
    raw_df   : pandas DataFrame
        The same raw data you passed to `fit()`.

    Returns
    -------
    float     – overall silhouette score (range: –1 … +1).
    """
    # 1 ─ get the cluster labels
    try:
        labels = pipeline[-1].labels_          # most sklearn clusterers
    except AttributeError:
        labels = pipeline[-1].predict(         # e.g. GaussianMixture
            pipeline[:-1].transform(raw_df)
        )

    # 2 ─ obtain the numeric array that the clusterer used
    #     (everything before the final step in the pipeline)
    X = pipeline[:-1].transform(raw_df)

    # 3 ─ compute silhouette
    return silhouette_score(X, labels)

# ───────────────────────── usage demo ─────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/HR_data.csv") 
    df_orgnl = pd.read_csv("data/HR_data.csv")        # adjust path as needed
    pipe = build_pipeline(df, n_clusters=3) 

    labels = pipe.fit_predict(df)               # fit + cluster
    df["cluster"] = labels
    sil = compute_silhouette(pipe, df_orgnl)

    print(df[["cluster"]].value_counts().sort_index())
    print(f"silhouette = {sil:.3f}")    # print silhouttescore
