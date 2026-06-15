import argparse

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

from project_config import (
    add_config_argument,
    load_clean_dataset,
    load_config,
    output_path,
    write_experiment_manifest,
)

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline model experiments.")
    add_config_argument(parser)
    return parser.parse_args()


def build_default_models(random_seed, n_jobs):
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
                ("model", LogisticRegression(max_iter=5000, random_state=random_seed)),
            ]
        ),
        "decision_tree_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(random_state=random_seed)),
            ]
        ),
        "random_forest_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=random_seed, n_jobs=n_jobs)),
            ]
        ),
        "gradient_boosting_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=random_seed)),
            ]
        ),
    }


def prefixed_grid(parameters):
    return {f"model__{name}": values for name, values in parameters.items()}


def build_tuned_models(config):
    random_seed = config["random_seed"]
    n_jobs = config["n_jobs"]
    model_config = config["baseline_models"]
    return {
        "logistic_regression_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, random_state=random_seed)),
                ]
            ),
            "params": prefixed_grid(model_config["logistic_regression"]),
        },
        "decision_tree_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", DecisionTreeClassifier(random_state=random_seed)),
                ]
            ),
            "params": prefixed_grid(model_config["decision_tree"]),
        },
        "random_forest_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(random_state=random_seed, n_jobs=n_jobs)),
                ]
            ),
            "params": prefixed_grid(model_config["random_forest"]),
        },
        "gradient_boosting_tuned": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingClassifier(random_state=random_seed)),
                ]
            ),
            "params": prefixed_grid(model_config["gradient_boosting"]),
        },
    }


def summarize_cv_scores(cv_scores):
    summary = {}
    for metric_name in SCORING:
        values = cv_scores[f"test_{metric_name}"]
        summary[f"cv_{metric_name}_mean"] = values.mean()
        summary[f"cv_{metric_name}_std"] = values.std()
    return summary


def evaluate_estimator(
    name,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    cv,
    n_jobs,
    config_type,
    best_params=None,
):
    cv_scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=n_jobs)
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
    args = parse_args()
    config = load_config(args.config)
    random_seed = config["random_seed"]
    n_jobs = config["n_jobs"]
    X, y, duplicate_count = load_clean_dataset(config)
    print(f"Dataset after duplicate removal: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Removed exact duplicate rows: {duplicate_count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        stratify=y,
        random_state=random_seed,
    )
    cv = StratifiedKFold(
        n_splits=config["cv_splits"],
        shuffle=True,
        random_state=random_seed,
    )

    results = []
    reports = []
    predictions_by_model = {}
    estimators_by_model = {}

    for name, estimator in build_default_models(random_seed, n_jobs).items():
        result, predictions, fitted_estimator = evaluate_estimator(
            name, estimator, X_train, X_test, y_train, y_test, cv, n_jobs, "default"
        )
        results.append(result)
        predictions_by_model[name] = predictions
        estimators_by_model[name] = fitted_estimator
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        reports.append(pd.DataFrame(report).transpose().assign(model=name))
        print(f"{name}: test_accuracy={result['test_accuracy']:.4f}")

    for name, experiment in build_tuned_models(config).items():
        grid = GridSearchCV(
            experiment["pipeline"],
            experiment["params"],
            cv=cv,
            scoring=config["primary_metric"],
            n_jobs=n_jobs,
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
            n_jobs,
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
    results_path = output_path(config, "assignment7_baseline_results.csv")
    reports_path = output_path(config, "assignment7_classification_reports.csv")
    confusion_matrix_path = output_path(config, "assignment7_best_confusion_matrix.csv")
    results_df.to_csv(results_path, index=False)
    pd.concat(reports).to_csv(reports_path)

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
    cm_df.to_csv(confusion_matrix_path)

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
    print(f"\nSaved results to: {results_path}")
    print(f"Saved classification reports to: {reports_path}")
    print(f"Saved best-model confusion matrix to: {confusion_matrix_path}")
    print(f"Updated manifest: {write_experiment_manifest(config, 'baseline_models')}")


if __name__ == "__main__":
    main()
