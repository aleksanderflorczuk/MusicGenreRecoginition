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

print(f"Dataset shape: {df.shape}")

X = df.drop(columns="genre")
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=30
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])

param_grid_svm = {
    "model__C": [1, 10, 50],
    "model__gamma": ["scale", 0.01, 0.001],
    "model__kernel": ["rbf"]
}

grid_svm = GridSearchCV(pipeline, param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1)
grid_svm.fit(X_train, y_train)

best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print("\n Best SVM Parameters:", grid_svm.best_params_)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_svm.named_steps["model"].classes_,
            yticklabels=best_svm.named_steps["model"].classes_)
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
