from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "features_extended.csv"
DOCS_DIR = PROJECT_ROOT / "docs"

TARGET_RESULTS_PATH = DOCS_DIR / "assignment8_target_results.csv"
COMPARATIVE_RESULTS_PATH = DOCS_DIR / "assignment8_comparative_results.csv"
OPTIMIZATION_RESULTS_PATH = DOCS_DIR / "assignment8_optimization_results.csv"
LEARNING_CURVE_PATH = DOCS_DIR / "assignment8_learning_curve.csv"
CLASSIFICATION_REPORT_PATH = DOCS_DIR / "assignment8_classification_report.csv"
CONFUSION_MATRIX_PATH = DOCS_DIR / "assignment8_best_confusion_matrix.csv"

RANDOM_STATE = 30
TEST_SIZE = 0.2
CV_SPLITS = 5

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)

    missing_count = int(df.isna().sum().sum())
    if missing_count:
        raise ValueError(f"Dataset contains missing values: {missing_count}")

    X = df.drop(columns="genre")
    y = df["genre"]
    return X, y, duplicate_count


def build_target_experiments():
    return {
        "svm_rbf_all_features": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf")),
                ]
            ),
            "params": {
                "model__C": [1, 3, 10, 30, 100],
                "model__gamma": ["scale", 0.03, 0.01, 0.003, 0.001],
            },
        },
        "svm_rbf_select_k_best": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=f_classif)),
                    ("model", SVC(kernel="rbf")),
                ]
            ),
            "params": {
                "selector__k": [20, 30, 40, 50, "all"],
                "model__C": [1, 3, 10, 30, 100],
                "model__gamma": ["scale", 0.03, 0.01, 0.003, 0.001],
            },
        },
    }


def evaluate_baseline_reference(X_train, X_test, y_train, y_test, cv):
    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    start = perf_counter()
    cv_scores = GridSearchCV(
        baseline,
        param_grid={},
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )
    cv_scores.fit(X_train, y_train)
    fit_time = perf_counter() - start

    start = perf_counter()
    predictions = cv_scores.predict(X_test)
    predict_time = perf_counter() - start

    return {
        "model": "random_forest_default",
        "role": "best_assignment7_baseline",
        "cv_accuracy_mean": cv_scores.best_score_,
        "cv_accuracy_std": cv_scores.cv_results_["std_test_score"][0],
        "train_accuracy_mean": cv_scores.cv_results_["mean_train_score"][0],
        "train_validation_gap": cv_scores.cv_results_["mean_train_score"][0] - cv_scores.best_score_,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "fit_time_seconds": fit_time,
        "test_predict_time_seconds": predict_time,
        "best_params": "{}",
    }


def summarize_cv_results(grid):
    rows = []
    cv_results = pd.DataFrame(grid.cv_results_)
    for _, row in cv_results.iterrows():
        rows.append(
            {
                "experiment": grid.estimator_name,
                "params": row["params"],
                "mean_train_accuracy": row["mean_train_score"],
                "std_train_accuracy": row["std_train_score"],
                "mean_validation_accuracy": row["mean_test_score"],
                "std_validation_accuracy": row["std_test_score"],
                "generalization_gap": row["mean_train_score"] - row["mean_test_score"],
                "rank_validation_accuracy": row["rank_test_score"],
                "mean_fit_time_seconds": row["mean_fit_time"],
            }
        )
    return rows


def evaluate_best_model(name, grid, X_train, X_test, y_train, y_test):
    start = perf_counter()
    predictions = grid.predict(X_test)
    predict_time = perf_counter() - start

    best_index = grid.best_index_
    cv_results = grid.cv_results_
    return {
        "model": name,
        "configuration": "optimized",
        "best_params": grid.best_params_,
        "cv_accuracy_mean": grid.best_score_,
        "cv_accuracy_std": cv_results["std_test_score"][best_index],
        "train_accuracy_mean": cv_results["mean_train_score"][best_index],
        "train_validation_gap": cv_results["mean_train_score"][best_index] - grid.best_score_,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "grid_fit_time_seconds": grid.fit_time_seconds,
        "test_predict_time_seconds": predict_time,
    }, predictions


def save_learning_curve(estimator, X_train, y_train, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    curve = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_accuracy_mean": train_scores.mean(axis=1),
            "train_accuracy_std": train_scores.std(axis=1),
            "validation_accuracy_mean": validation_scores.mean(axis=1),
            "validation_accuracy_std": validation_scores.std(axis=1),
        }
    )
    curve["generalization_gap"] = curve["train_accuracy_mean"] - curve["validation_accuracy_mean"]
    curve.to_csv(LEARNING_CURVE_PATH, index=False)


def main():
    X, y, duplicate_count = load_dataset()
    print(f"Dataset after duplicate removal: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Removed exact duplicate rows: {duplicate_count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    target_results = []
    optimization_rows = []
    predictions_by_model = {}
    grids = {}

    for name, experiment in build_target_experiments().items():
        print(f"\nRunning target experiment: {name}")
        grid = GridSearchCV(
            experiment["pipeline"],
            experiment["params"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=True,
        )
        grid.estimator_name = name
        start = perf_counter()
        grid.fit(X_train, y_train)
        grid.fit_time_seconds = perf_counter() - start

        result, predictions = evaluate_best_model(name, grid, X_train, X_test, y_train, y_test)
        target_results.append(result)
        optimization_rows.extend(summarize_cv_results(grid))
        predictions_by_model[name] = predictions
        grids[name] = grid

        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        print(f"Test accuracy: {result['test_accuracy']:.4f}")
        print(f"Best params: {grid.best_params_}")

    results_df = pd.DataFrame(target_results).sort_values(
        by=["test_accuracy", "test_f1_macro"], ascending=False
    )
    results_df.to_csv(TARGET_RESULTS_PATH, index=False)

    baseline_reference = evaluate_baseline_reference(X_train, X_test, y_train, y_test, cv)
    best_target = results_df.iloc[0].to_dict()
    comparative_df = pd.DataFrame(
        [
            baseline_reference,
            {
                "model": best_target["model"],
                "role": "assignment8_primary_model",
                "cv_accuracy_mean": best_target["cv_accuracy_mean"],
                "cv_accuracy_std": best_target["cv_accuracy_std"],
                "train_accuracy_mean": best_target["train_accuracy_mean"],
                "train_validation_gap": best_target["train_validation_gap"],
                "test_accuracy": best_target["test_accuracy"],
                "test_precision_macro": best_target["test_precision_macro"],
                "test_recall_macro": best_target["test_recall_macro"],
                "test_f1_macro": best_target["test_f1_macro"],
                "fit_time_seconds": best_target["grid_fit_time_seconds"],
                "test_predict_time_seconds": best_target["test_predict_time_seconds"],
                "best_params": best_target["best_params"],
            },
        ]
    )
    baseline_accuracy = comparative_df.loc[
        comparative_df["role"] == "best_assignment7_baseline", "test_accuracy"
    ].iloc[0]
    baseline_f1 = comparative_df.loc[
        comparative_df["role"] == "best_assignment7_baseline", "test_f1_macro"
    ].iloc[0]
    comparative_df["absolute_accuracy_improvement_vs_baseline"] = (
        comparative_df["test_accuracy"] - baseline_accuracy
    )
    comparative_df["relative_accuracy_improvement_vs_baseline"] = (
        comparative_df["absolute_accuracy_improvement_vs_baseline"] / baseline_accuracy
    )
    comparative_df["absolute_f1_improvement_vs_baseline"] = comparative_df["test_f1_macro"] - baseline_f1
    comparative_df["relative_f1_improvement_vs_baseline"] = (
        comparative_df["absolute_f1_improvement_vs_baseline"] / baseline_f1
    )
    comparative_df.to_csv(COMPARATIVE_RESULTS_PATH, index=False)
    pd.DataFrame(optimization_rows).sort_values(
        by=["experiment", "rank_validation_accuracy", "generalization_gap"]
    ).to_csv(OPTIMIZATION_RESULTS_PATH, index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_grid = grids[best_model_name]
    best_predictions = predictions_by_model[best_model_name]

    report = classification_report(y_test, best_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_PATH)

    labels = best_grid.best_estimator_.named_steps["model"].classes_
    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(CONFUSION_MATRIX_PATH)
    save_learning_curve(best_grid.best_estimator_, X_train, y_train, cv)

    print("\nRanking by test accuracy:")
    print(
        results_df[
            [
                "model",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "train_accuracy_mean",
                "train_validation_gap",
                "test_accuracy",
                "test_f1_macro",
                "best_params",
                "grid_fit_time_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved target results to: {TARGET_RESULTS_PATH}")
    print(f"Saved comparative results to: {COMPARATIVE_RESULTS_PATH}")
    print(f"Saved optimization results to: {OPTIMIZATION_RESULTS_PATH}")
    print(f"Saved learning curve to: {LEARNING_CURVE_PATH}")
    print(f"Saved classification report to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
