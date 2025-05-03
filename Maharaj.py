from typing    import Tuple, List
import pandas   as pd
import numpy as np
import warnings

# Monkey-patch so sklearn’s np.warnings refers to the real warnings module
np.warnings = warnings

from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster       import KMeans, AgglomerativeClustering
from sklearn.metrics       import silhouette_score

RANDOM_STATE = 42   # for reproducibility


# e.g. for a 5-point Likert:

def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    emotion_words = {
        "frustrated", "upset", "hostile", "alert", "ashamed",
        "inspired", "nervous", "attentive", "afraid", "active", "determined",
    }
    num_cols, emo_cols = [], []
    for c in df.select_dtypes("number").columns:
        if c.lower() in emotion_words:
            emo_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, emo_cols

def build_pipeline(df: pd.DataFrame,
                   n_clusters: int = 4,
                   pca_var: float = 0.90,
                  ) -> Pipeline:
    """
    Now uses a robust pipeline for numeric features:
      impute → power-transform → robust-scale
    Emotion ratings still impute → standard-scale.
    """
    num_cols, emo_cols = split_columns(df)

    # ── Numeric features: handle skew/outliers robustly ────────────────
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("power",  PowerTransformer(method="yeo-johnson")),
        ("robust", RobustScaler()),
    ])

    # ── Emotion ratings: simple z-scoring ───────────────────────────────
    emo_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler()),
    ])


    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("emo", emo_pipe, emo_cols),
    ], remainder="drop")

    steps = [("pre", pre)]
    if pca_var:
        steps.append((
            "pca",
            PCA(n_components=pca_var,
                svd_solver="full",
                random_state=RANDOM_STATE)
        ))
    # ← Hierarchical clustering step instead of KMeans
    steps.append((
        "cluster",
        AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="ward"
        )
    ))
    return Pipeline(steps)

def compute_silhouette(pipe: Pipeline, raw_df: pd.DataFrame) -> float:
    # extract labels
    try:
        labels = pipe[-1].labels_
    except AttributeError:
        labels = pipe[-1].predict(pipe[:-1].transform(raw_df))
    # get the whitened/robust‐scaled feature space
    X = pipe[:-1].transform(raw_df)
    return silhouette_score(X, labels)

if __name__ == "__main__":
    df      = pd.read_csv("data/HR_data.csv")
    df_raw  = df.copy()

    pipe    = build_pipeline(df, n_clusters=3, pca_var=0.7)
    labels  = pipe.fit_predict(df)
    df["cluster"] = labels

    print("Cluster counts:\n", df["cluster"].value_counts().sort_index(), "\n")
    sil = compute_silhouette(pipe, df_raw)
    print(f"Silhouette score: {sil:.3f}")

    continuous_cols = df.columns[:52].tolist()
    string_cat_cols = [df.columns[52], df.columns[53], df.columns[57]]
    numeric_cat_cols = [col for col in df.columns[52:68] if col not in string_cat_cols]
    categorical_cols = string_cat_cols + numeric_cat_cols
    emotion_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                    'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']

    # --- Impute continuous, numeric categorical, and emotion features ---
    imputer_cont = SimpleImputer(strategy='mean')
    imputer_ordinal = SimpleImputer(strategy='median')

    df[continuous_cols] = imputer_cont.fit_transform(df[continuous_cols])
    df[numeric_cat_cols] = imputer_ordinal.fit_transform(df[numeric_cat_cols])
    df[emotion_cols] = imputer_ordinal.fit_transform(df[emotion_cols])

    # --- Clip emotion values to known scales before scaling ---
    df['Frustrated'] = df['Frustrated'].clip(lower=1, upper=10)
    for col in emotion_cols:
        if col != 'Frustrated':
            df[col] = df[col].clip(lower=1, upper=5)

    # --- Normalize continuous and numeric categorical features ---
    scaler = RobustScaler()
    transformer = PowerTransformer(method="yeo-johnson")
    df[continuous_cols] = transformer.fit_transform(df[continuous_cols])
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])


