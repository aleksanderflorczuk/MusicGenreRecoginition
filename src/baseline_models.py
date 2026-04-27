from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "features_extended.csv"
RESULTS_PATH = PROJECT_ROOT / "docs" / "assignment7_baseline_results.csv"
REPORTS_PATH = PROJECT_ROOT / "docs" / "assignment7_classification_reports.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "docs" / "assignment7_best_confusion_matrix.csv"

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


def build_default_models():
    return {
        "dummy_most_frequent_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "decision_tree_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def build_tuned_models():
    return {
        "logistic_regression_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
                ]
            ),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
            },
        },
        "decision_tree_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "params": {
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "random_forest_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            "params": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 15],
                "model__max_features": ["sqrt", "log2"],
            },
        },
        "gradient_boosting_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        },
    }


def summarize_cv_scores(cv_scores):
    summary = {}
    for metric_name in SCORING:
        values = cv_scores[f"test_{metric_name}"]
        summary[f"cv_{metric_name}_mean"] = values.mean()
        summary[f"cv_{metric_name}_std"] = values.std()
    return summary


def evaluate_estimator(name, estimator, X_train, X_test, y_train, y_test, cv, config_type, best_params=None):
    cv_scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    result = {
        "model": name,
        "configuration": config_type,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "best_params": best_params or "{}",
    }
    result.update(summarize_cv_scores(cv_scores))
    return result, predictions, estimator


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

    results = []
    reports = []
    predictions_by_model = {}
    estimators_by_model = {}

    for name, estimator in build_default_models().items():
        result, predictions, fitted_estimator = evaluate_estimator(
            name, estimator, X_train, X_test, y_train, y_test, cv, "default"
        )
        results.append(result)
        predictions_by_model[name] = predictions
        estimators_by_model[name] = fitted_estimator
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        reports.append(pd.DataFrame(report).transpose().assign(model=name))
        print(f"{name}: test_accuracy={result['test_accuracy']:.4f}")

    for name, experiment in build_tuned_models().items():
        grid = GridSearchCV(
            experiment["pipeline"],
            experiment["params"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        result, predictions, fitted_estimator = evaluate_estimator(
            name,
            grid.best_estimator_,
            X_train,
            X_test,
            y_train,
            y_test,
            cv,
            "tuned",
            grid.best_params_,
        )
        results.append(result)
        predictions_by_model[name] = predictions
        estimators_by_model[name] = fitted_estimator
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        reports.append(pd.DataFrame(report).transpose().assign(model=name))
        print(f"{name}: test_accuracy={result['test_accuracy']:.4f}, best_params={grid.best_params_}")

    results_df = pd.DataFrame(results).sort_values(
        by=["test_accuracy", "test_f1_macro"], ascending=False
    )
    results_df.to_csv(RESULTS_PATH, index=False)
    pd.concat(reports).to_csv(REPORTS_PATH)

    best_model_name = results_df.iloc[0]["model"]
    best_estimator = estimators_by_model[best_model_name]
    cm = confusion_matrix(
        y_test,
        predictions_by_model[best_model_name],
        labels=best_estimator.named_steps["model"].classes_,
    )
    cm_df = pd.DataFrame(
        cm,
        index=best_estimator.named_steps["model"].classes_,
        columns=best_estimator.named_steps["model"].classes_,
    )
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    print("\nRanking by test accuracy:")
    print(
        results_df[
            [
                "model",
                "configuration",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "test_accuracy",
                "test_f1_macro",
                "best_params",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to: {RESULTS_PATH}")
    print(f"Saved classification reports to: {REPORTS_PATH}")
    print(f"Saved best-model confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
