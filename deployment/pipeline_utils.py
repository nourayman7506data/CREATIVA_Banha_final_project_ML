# ============================================================
# pipeline_utils.py
# ------------------------------------------------------------
# الكلاس المخصص (IQRCapper) لازم يكون في ملف منفصل عشان يقدر
# joblib/pickle يفتح الموديل صح من أي ملف (train_model.py أو app.py)
# ============================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.q1_ = X.quantile(0.25)
        self.q3_ = X.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_bounds_ = self.q1_ - self.factor * self.iqr_
        self.upper_bounds_ = self.q3_ + self.factor * self.iqr_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X.values
