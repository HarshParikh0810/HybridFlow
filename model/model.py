import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


df=pd.read_csv("D:\Dataset_featuring\Final_dataset.csv")
df.drop(["dtype","cpu_runtime_ms","gpu_runtime_ms","transfer_time_ms","power_mode","thermal_headroom","device_name","is_edge","gpu_mem_used_mb","gpu_mem_total_mb","gpu_mem_free_mb","source_file","concurrent_gpu_tasks","data_transfer_cost_ms"],axis=1,inplace=True)
knn_imputer = KNNImputer(n_neighbors=8)
preprocessor = ColumnTransformer(
    transformers=[
          
        ('gpu_mem_impute', KNNImputer(n_neighbors=3), [5]),             
        ('op_type_encode', OneHotEncoder(handle_unknown='ignore'), [0]) 
    ],
    remainder='passthrough'  
)

X=df.drop("winner",axis=1)
y=df["winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]), 
    verbosity=0
)

pipeline_xgb = ImbPipeline(steps=[
    ('preprocessing', preprocessor),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb)
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipeline_xgb, X, y, cv=skf)

cv_scores = cross_val_score(pipeline_xgb, X, y, cv=skf, scoring='accuracy')

print("Fold Accuracies:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))

print("XGBoost Classification Report:\n", classification_report(y, y_pred))

pipeline_xgb.fit(X_train, y_train)

joblib.dump(pipeline_xgb, "preprocesing_model.pkl")
print("Model saved as preprocessing_model.pkl")
