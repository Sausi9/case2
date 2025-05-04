from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class FeatureConcatenator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, Xs):
        return np.hstack(Xs)

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

def make_custom_preprocessing_pipeline(nominal_categorical_features, ordinal_categorical_features, binary_categorical_features, continuous_features, emotion_features, drop_features=None, n_pca_components=7, use_pca=True):
    """
    Returns a preprocessing pipeline for continuous, ordinal, nominal, binary, and emotion features.
    """
    drop_features = drop_features or []

    # Drop unwanted columns
    dropper = ('drop', DropColumns(columns=drop_features), drop_features)

    # Continuous features: impute, scale, optional PCA
    if use_pca:
        continuous_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components))
        ])
    else:
        continuous_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ])

    # Ordinal categorical features: impute, scale
    ordinal_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('scale', MinMaxScaler())
    ])

    # Nominal categorical features: one-hot encode
    nominal_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Binary features: passthrough after imputation
    binary_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent'))
    ])

    # Emotion features: clip + scale using custom transformer
    class EmotionClipScale(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scalers = {}
            self.imputer = SimpleImputer(strategy='most_frequent')

        def fit(self, X, y=None):
            X = pd.DataFrame(X)
            X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

            for col in X_imputed.columns:
                if col == 'Frustrated':
                    self.scalers[col] = MinMaxScaler().fit([[1], [10]])
                else:
                    self.scalers[col] = MinMaxScaler().fit([[1], [5]])
            return self

        def transform(self, X):
            X = pd.DataFrame(X).copy()
            X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)

            for col in X_imputed.columns:
                if col == 'Frustrated':
                    X_imputed[col] = X_imputed[col].clip(lower=1, upper=10)
                else:
                    X_imputed[col] = X_imputed[col].clip(lower=1, upper=5)
                X_imputed[[col]] = self.scalers[col].transform(X_imputed[[col]])

            return X_imputed.values

    emotion_pipeline = Pipeline([
        ('clipscale', EmotionClipScale())
    ])

    column_transformer = ColumnTransformer(transformers=[
        dropper,
        ('cont', continuous_pipeline, continuous_features),
        ('ord', ordinal_pipeline, ordinal_categorical_features),
        ('nom', nominal_pipeline, nominal_categorical_features),
        ('bin', binary_pipeline, binary_categorical_features),
        ('emo', emotion_pipeline, emotion_features)
    ])

    return Pipeline([
        ('transform', column_transformer)
    ])

def load_and_preprocess_data(file_path, n_pca_components=7, use_pca=True):
    data = pd.read_csv(file_path).iloc[:, 1:]
    continuous_cols = data.columns[:51].tolist()
    nominal_cat_cols = [data.columns[51], data.columns[52], data.columns[56]]
    ordinal_cat_cols = [col for col in data.columns[51:67] if col not in nominal_cat_cols and col != 'Puzzler']
    binary_cat_cols = ['Puzzler'] if 'Puzzler' in data.columns else []
    drop_cols = ['Individual'] if 'Individual' in data.columns else []
    emotion_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                    'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']

    pipeline = make_custom_preprocessing_pipeline(
        nominal_categorical_features=nominal_cat_cols,
        ordinal_categorical_features=ordinal_cat_cols,
        binary_categorical_features=binary_cat_cols,
        continuous_features=continuous_cols,
        emotion_features=emotion_cols,
        drop_features=drop_cols,
        n_pca_components=n_pca_components,
        use_pca=use_pca
    )

    transformed_data = pipeline.fit_transform(data)

    # Compute feature flags
    ohe_nom = pipeline.named_steps['transform'].named_transformers_['nom']['onehot']
    ohe_nom_dim = len(ohe_nom.get_feature_names_out())
    n_cont = n_pca_components if use_pca else len(continuous_cols)
    n_ordinal = len(ordinal_cat_cols)
    n_binary = len(binary_cat_cols)
    n_emotion = len(emotion_cols)

    cat_feature_flags = [False] * n_cont + [True] * (n_ordinal + ohe_nom_dim + n_binary + n_emotion)

    return data, pd.DataFrame(transformed_data, index=data.index), nominal_cat_cols, ordinal_cat_cols, binary_cat_cols, emotion_cols, cat_feature_flags
