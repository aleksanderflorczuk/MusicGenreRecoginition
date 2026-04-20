from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "features_extended.csv"
df = pd.read_csv(DATA_PATH)

duplicate_count = df.duplicated().sum()
if duplicate_count:
    print(f"Removing exact duplicate rows: {duplicate_count}")
    df = df.drop_duplicates().reset_index(drop=True)

missing_count = df.isna().sum().sum()
if missing_count:
    raise ValueError(f"Dataset contains missing values: {missing_count}")

SELECTED_FEATURES = [
    "tempo",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_std",
    "spectral_rolloff_std",
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "mfcc_4_mean",
    "mfcc_5_mean",
    "mfcc_6_mean",
    "mfcc_7_mean",
    "mfcc_8_mean",
    "mfcc_9_mean",
    "mfcc_10_mean",
    "mfcc_1_std",
    "mfcc_2_std",
    "mfcc_3_std",
    "mfcc_4_std",
    "mfcc_5_std",
    "mfcc_6_std",
    "mfcc_7_std",
    "chroma_1_mean",
    "chroma_5_mean",
    "chroma_3_mean",
    "chroma_1_std",
    "chroma_5_std",
]

X = df[SELECTED_FEATURES]
y = df["genre"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=30
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf"))
])

param_grid = {
    "model__C": [1, 10, 50],
    "model__gamma": ["scale", 0.01, 0.001]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)


print("Best parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=best_svm.named_steps["model"].classes_,
            yticklabels=best_svm.named_steps["model"].classes_)
plt.title("Confusion Matrix - SVM (Selected Features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
