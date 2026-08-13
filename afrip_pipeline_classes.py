"""Custom preprocessing classes required by the serialized fraud model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["dob"] = pd.to_datetime(df["dob"])
        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_day"] = df["trans_date_trans_time"].dt.day
        df["trans_month"] = df["trans_date_trans_time"].dt.month
        df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
        df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25).astype(int)
        df["distance_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
        return df


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy().drop(columns=self.columns, errors="ignore")


class GenderEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["gender"] = df["gender"].map({"F": 0, "M": 1})
        return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.frequency_maps_ = {col: X[col].value_counts(normalize=True) for col in self.columns}
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.columns:
            df[f"{col}_freq"] = df[col].map(self.frequency_maps_[col]).fillna(0)
        return df


class FinalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, onehot_cols):
        self.numeric_cols = numeric_cols
        self.onehot_cols = onehot_cols

    def fit(self, X, y=None):
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe_.fit(X[self.onehot_cols])
        return self

    def transform(self, X):
        df = X.copy()
        ohe_data = self.ohe_.transform(df[self.onehot_cols])
        ohe_df = pd.DataFrame(ohe_data, columns=self.ohe_.get_feature_names_out(self.onehot_cols), index=df.index)
        return pd.concat([df[self.numeric_cols + ["gender", "merchant_freq", "job_freq"]], ohe_df], axis=1)


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.85):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def get_feature_importances(self):
        return self.model.feature_importances_
