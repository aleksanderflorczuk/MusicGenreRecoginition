import argparse

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from project_config import (
    add_config_argument,
    load_clean_dataset,
    load_config,
    output_path,
    write_experiment_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final result analysis.")
    add_config_argument(parser)
    return parser.parse_args()


def save_consolidated_results(paths):
    baseline = pd.read_csv(paths["baseline_results"])
    baseline = baseline.assign(source="assignment7_baseline")

    target = pd.read_csv(paths["target_results"])
    target = target.assign(source="assignment8_primary")

    common_columns = [
        "source",
        "model",
        "configuration",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "best_params",
    ]
    consolidated = pd.concat(
        [baseline[common_columns], target[common_columns]],
        ignore_index=True,
    ).sort_values(by=["test_accuracy", "test_f1_macro"], ascending=False)
    consolidated.to_csv(paths["consolidated_results"], index=False)


def save_error_analysis(paths, y_test, predictions, labels):
    cm = confusion_matrix(y_test, predictions, labels=labels)
    rows = []
    pair_rows = []

    for index, label in enumerate(labels):
        support = int(cm[index].sum())
        correct = int(cm[index, index])
        incorrect = support - correct
        recall = correct / support if support else 0.0

        false_positives = int(cm[:, index].sum() - correct)
        predicted_as_label = int(cm[:, index].sum())
        precision = correct / predicted_as_label if predicted_as_label else 0.0

        rows.append(
            {
                "class": label,
                "support": support,
                "correct": correct,
                "incorrect": incorrect,
                "recall": recall,
                "precision": precision,
                "false_positives": false_positives,
            }
        )

        for col_index, predicted_label in enumerate(labels):
            if index == col_index:
                continue
            count = int(cm[index, col_index])
            if count:
                pair_rows.append(
                    {
                        "true_class": label,
                        "predicted_class": predicted_label,
                        "count": count,
                    }
                )

    pd.DataFrame(rows).sort_values(by=["incorrect", "false_positives"], ascending=False).to_csv(
        paths["error_summary"],
        index=False,
    )
    pd.DataFrame(pair_rows).sort_values(by="count", ascending=False).to_csv(
        paths["misclassification_pairs"],
        index=False,
    )


def save_permutation_importance(config, paths, model, X_test, y_test):
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=config["analysis"]["permutation_repeats"],
        random_state=config["random_seed"],
        scoring=config["primary_metric"],
        n_jobs=config["n_jobs"],
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    importance_df.to_csv(paths["feature_importance"], index=False)


def save_hyperparameter_stability(config, paths):
    optimization = pd.read_csv(paths["optimization_results"])
    optimization = optimization[optimization["experiment"] == "svm_rbf_all_features"].copy()
    optimization = optimization.sort_values(
        by=["rank_validation_accuracy", "generalization_gap"]
    )
    optimization.head(config["analysis"]["stability_rows"]).to_csv(
        paths["hyperparameter_stability"], index=False
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = {
        "baseline_results": output_path(config, "assignment7_baseline_results.csv"),
        "target_results": output_path(config, "assignment8_target_results.csv"),
        "optimization_results": output_path(config, "assignment8_optimization_results.csv"),
        "consolidated_results": output_path(config, "assignment9_consolidated_results.csv"),
        "error_summary": output_path(config, "assignment9_error_summary.csv"),
        "misclassification_pairs": output_path(
            config, "assignment9_misclassification_pairs.csv"
        ),
        "feature_importance": output_path(
            config, "assignment9_permutation_importance.csv"
        ),
        "hyperparameter_stability": output_path(
            config, "assignment9_hyperparameter_stability.csv"
        ),
    }
    X, y, _ = load_clean_dataset(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        stratify=y,
        random_state=config["random_seed"],
    )

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel=config["target_model"]["kernel"],
                    C=config["analysis"]["final_C"],
                    gamma=config["analysis"]["final_gamma"],
                ),
            ),
        ]
    )
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)
    labels = final_model.named_steps["model"].classes_

    save_consolidated_results(paths)
    save_error_analysis(paths, y_test, predictions, labels)
    save_permutation_importance(config, paths, final_model, X_test, y_test)
    save_hyperparameter_stability(config, paths)

    print(f"Saved consolidated results to: {paths['consolidated_results']}")
    print(f"Saved error summary to: {paths['error_summary']}")
    print(f"Saved misclassification pairs to: {paths['misclassification_pairs']}")
    print(f"Saved permutation importance to: {paths['feature_importance']}")
    print(
        "Saved hyperparameter stability table to: "
        f"{paths['hyperparameter_stability']}"
    )
    print(f"Updated manifest: {write_experiment_manifest(config, 'results_analysis')}")


if __name__ == "__main__":
    main()
