# from pathlib import Path
#
# import pandas as pd
# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
#
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DATA_PATH = PROJECT_ROOT / "data" / "features_extended.csv"
# RANDOM_STATE = 30
#
#
# def load_data():
#     df = pd.read_csv(DATA_PATH)
#
#     duplicate_count = df.duplicated().sum()
#     if duplicate_count:
#         print(f"Removing exact duplicate rows: {duplicate_count}")
#         df = df.drop_duplicates().reset_index(drop=True)
#
#     missing_count = df.isna().sum().sum()
#     if missing_count:
#         raise ValueError(f"Dataset contains missing values: {missing_count}")
#
#     X = df.drop(columns="genre")
#     y = df["genre"]
#     return X, y
#
#
# def build_experiments():
#     return {
#         "svm_rbf_all_features": {
#             "pipeline": Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("model", SVC(kernel="rbf"))
#             ]),
#             "params": {
#                 "model__C": [1, 3, 10, 30, 100, 300],
#                 "model__gamma": ["scale", 0.03, 0.01, 0.003, 0.001],
#             },
#         },
#         "svm_rbf_select_k_best": {
#             "pipeline": Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("selector", SelectKBest(score_func=f_classif)),
#                 ("model", SVC(kernel="rbf"))
#             ]),
#             "params": {
#                 "selector__k": [20, 30, 40, 50, "all"],
#                 "model__C": [1, 3, 10, 30, 100, 300],
#                 "model__gamma": ["scale", 0.03, 0.01, 0.003, 0.001],
#             },
#         },
#         "knn_scaled": {
#             "pipeline": Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("model", KNeighborsClassifier())
#             ]),
#             "params": {
#                 "model__n_neighbors": [3, 5, 7, 9, 11, 15],
#                 "model__weights": ["uniform", "distance"],
#                 "model__p": [1, 2],
#             },
#         },
#         "random_forest": {
#             "pipeline": Pipeline([
#                 ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
#             ]),
#             "params": {
#                 "model__n_estimators": [300, 600],
#                 "model__max_depth": [None, 12, 20],
#                 "model__min_samples_split": [2, 5],
#                 "model__max_features": ["sqrt", "log2"],
#             },
#         },
#         "extra_trees": {
#             "pipeline": Pipeline([
#                 ("model", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1))
#             ]),
#             "params": {
#                 "model__n_estimators": [300, 600],
#                 "model__max_depth": [None, 12, 20],
#                 "model__min_samples_split": [2, 5],
#                 "model__max_features": ["sqrt", "log2"],
#             },
#         },
#     }
#
#
# def main():
#     X, y = load_data()
#     print(f"Dataset shape after cleaning: {X.shape}")
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         stratify=y,
#         random_state=RANDOM_STATE,
#     )
#
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#     experiments = build_experiments()
#     results = []
#
#     for name, experiment in experiments.items():
#         print(f"\nRunning: {name}")
#         grid = GridSearchCV(
#             experiment["pipeline"],
#             param_grid=experiment["params"],
#             cv=cv,
#             scoring="accuracy",
#             n_jobs=-1,
#         )
#         grid.fit(X_train, y_train)
#
#         y_pred = grid.predict(X_test)
#         test_accuracy = accuracy_score(y_test, y_pred)
#
#         results.append({
#             "name": name,
#             "cv_accuracy": grid.best_score_,
#             "test_accuracy": test_accuracy,
#             "best_params": grid.best_params_,
#             "estimator": grid.best_estimator_,
#             "predictions": y_pred,
#         })
#
#         print(f"Best CV accuracy: {grid.best_score_:.4f}")
#         print(f"Test accuracy: {test_accuracy:.4f}")
#         print(f"Best params: {grid.best_params_}")
#
#     results = sorted(results, key=lambda item: item["test_accuracy"], reverse=True)
#     best = results[0]
#
#     print("\n=== Ranking by test accuracy ===")
#     for index, result in enumerate(results, start=1):
#         print(
#             f"{index}. {result['name']}: "
#             f"test={result['test_accuracy']:.4f}, "
#             f"cv={result['cv_accuracy']:.4f}"
#         )
#
#     print("\n=== Best model ===")
#     print(f"Model: {best['name']}")
#     print(f"Test accuracy: {best['test_accuracy']:.4f}")
#     print(f"Best params: {best['best_params']}")
#     print("\nClassification report:")
#     print(classification_report(y_test, best["predictions"]))
#     print("Confusion matrix:")
#     print(confusion_matrix(y_test, best["predictions"]))
#
#
# if __name__ == "__main__":
#     main()
