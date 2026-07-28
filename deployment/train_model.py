# ============================================================
# train_model.py
# ------------------------------------------------------------
# ده سكريبت التدريب: بياخد bank.csv وبيعمل بالظبط نفس خطوات
# التنظيف والـ preprocessing اللي في النوت بوك الأصلي
# (Building Full Pipeline) لكن بموديل LightGBM بدل RandomForest
# لأنه طلع أفضل موديل في مقارنة الـ accuracy.
#
# الناتج: ملف model_pipeline.pkl فيه كل حاجة (تنظيف + تشفير
# + اختيار فيتشرز + الموديل) في كيان واحد جاهز للـ deployment.
#
# طريقة التشغيل:
#   python train_model.py
# ============================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.feature_selection import SelectPercentile, f_classif
import category_encoders as ce
from lightgbm import LGBMClassifier

# الكلاس المخصص لازم يتحمل من ملف منفصل عشان الموديل يفتح صح
# في أي ملف تاني (زي app.py) من غير AttributeError
from pipeline_utils import IQRCapper


# ============================================================
# 1) قراءة البيانات
# ============================================================

df = pd.read_csv("bank.csv")

# استبدال unknown بـ NaN
df = df.replace("unknown", np.nan)

# حذف التكرار
df = df.drop_duplicates()

# التأكد من صحة عمود التارجت وتحويله لأرقام
df = df[df["deposit"].isin(["yes", "no"])].copy()
df["deposit"] = df["deposit"].map({"no": 0, "yes": 1})

X = df.drop("deposit", axis=1)
y = df["deposit"]


# ============================================================
# 2) تجميع الأعمدة حسب نوع المعالجة (نفس تقسيم النوت بوك)
# ============================================================

median_cols = ["balance", "duration", "previous", "pdays"]
mean_cols = ["age", "day"]
target_encoding_cols = ["job"]
education_cols = ["education"]
month_cols = ["month"]
onehot_cols = ["marital"]
minmax_categorical_cols = ["contact", "poutcome"]
binary_cols = ["default", "housing", "loan"]


# ============================================================
# 4) بناء الـ Pipelines الفرعية لكل مجموعة أعمدة
# ============================================================

median_pipeline = Pipeline(steps=[
    ("median_imputer", SimpleImputer(strategy="median")),
    ("outlier_capping", IQRCapper()),
    ("robust_scaler", RobustScaler()),
])

mean_standard_pipeline = Pipeline(steps=[
    ("mean_imputer", SimpleImputer(strategy="mean")),
    ("standard_scaler", StandardScaler()),
])

job_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("target_encoder", ce.TargetEncoder(smoothing=5)),
    ("minmax_scaler", MinMaxScaler(feature_range=(0, 1))),
])

education_order = ["primary", "secondary", "tertiary"]
education_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal_encoder", OrdinalEncoder(
        categories=[education_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
    ("minmax_scaler", MinMaxScaler(feature_range=(0, 1))),
])

months_order = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]
month_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("month_ordinal_encoder", OrdinalEncoder(
        categories=[months_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
    ("minmax_scaler", MinMaxScaler(feature_range=(0, 1))),
])

onehot_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

categorical_minmax_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ("minmax_scaler", MinMaxScaler(feature_range=(0, 1))),
])

binary_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])


# ============================================================
# 5) دمج كل الـ Pipelines في ColumnTransformer واحد
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("median_processing", median_pipeline, median_cols),
        ("mean_standard_processing", mean_standard_pipeline, mean_cols),
        ("job_target_encoding", job_pipeline, target_encoding_cols),
        ("education_encoding", education_pipeline, education_cols),
        ("month_encoding", month_pipeline, month_cols),
        ("marital_onehot_encoding", onehot_pipeline, onehot_cols),
        ("categorical_minmax_processing", categorical_minmax_pipeline, minmax_categorical_cols),
        ("binary_encoding", binary_pipeline, binary_cols),
    ],
    remainder="drop",
)


# ============================================================
# 6) الـ Pipeline الكامل: Preprocessing -> Feature Selection -> LightGBM
# ============================================================

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectPercentile(score_func=f_classif, percentile=50)),
    ("model", LGBMClassifier(random_state=1, verbose=-1)),
])


# ============================================================
# 7) التدريب على كل البيانات
# ============================================================

print("جاري تدريب الموديل...")
full_pipeline.fit(X, y)
print("تم التدريب بنجاح ✅")

train_accuracy = full_pipeline.score(X, y)
print(f"Train Accuracy: {train_accuracy:.4f}")


# ============================================================
# 8) حفظ الـ Pipeline كامل في ملف واحد
# ============================================================

joblib.dump(full_pipeline, "model_pipeline.pkl")
print("تم حفظ الموديل في model_pipeline.pkl ✅")
