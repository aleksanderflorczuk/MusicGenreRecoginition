from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "features_extended.csv"
DOCS_DIR = PROJECT_ROOT / "docs"

BASELINE_RESULTS_PATH = DOCS_DIR / "assignment7_baseline_results.csv"
TARGET_RESULTS_PATH = DOCS_DIR / "assignment8_target_results.csv"
OPTIMIZATION_RESULTS_PATH = DOCS_DIR / "assignment8_optimization_results.csv"

CONSOLIDATED_RESULTS_PATH = DOCS_DIR / "assignment9_consolidated_results.csv"
ERROR_SUMMARY_PATH = DOCS_DIR / "assignment9_error_summary.csv"
MISCLASSIFICATION_PAIRS_PATH = DOCS_DIR / "assignment9_misclassification_pairs.csv"
FEATURE_IMPORTANCE_PATH = DOCS_DIR / "assignment9_permutation_importance.csv"
HYPERPARAMETER_STABILITY_PATH = DOCS_DIR / "assignment9_hyperparameter_stability.csv"

RANDOM_STATE = 30
TEST_SIZE = 0.2


def load_clean_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    if df.isna().sum().sum():
        raise ValueError("Dataset contains missing values.")
    return df.drop(columns="genre"), df["genre"]


def save_consolidated_results():
    baseline = pd.read_csv(BASELINE_RESULTS_PATH)
    baseline = baseline.assign(source="assignment7_baseline")

    target = pd.read_csv(TARGET_RESULTS_PATH)
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
    consolidated.to_csv(CONSOLIDATED_RESULTS_PATH, index=False)


def save_error_analysis(y_test, predictions, labels):
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
        ERROR_SUMMARY_PATH,
        index=False,
    )
    pd.DataFrame(pair_rows).sort_values(by="count", ascending=False).to_csv(
        MISCLASSIFICATION_PAIRS_PATH,
        index=False,
    )


def save_permutation_importance(model, X_test, y_test):
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="accuracy",
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def save_hyperparameter_stability():
    optimization = pd.read_csv(OPTIMIZATION_RESULTS_PATH)
    optimization = optimization[optimization["experiment"] == "svm_rbf_all_features"].copy()
    optimization = optimization.sort_values(
        by=["rank_validation_accuracy", "generalization_gap"]
    )
    optimization.head(15).to_csv(HYPERPARAMETER_STABILITY_PATH, index=False)


def main():
    X, y = load_clean_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=3, gamma="scale")),
        ]
    )
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)
    labels = final_model.named_steps["model"].classes_

    save_consolidated_results()
    save_error_analysis(y_test, predictions, labels)
    save_permutation_importance(final_model, X_test, y_test)
    save_hyperparameter_stability()

    print(f"Saved consolidated results to: {CONSOLIDATED_RESULTS_PATH}")
    print(f"Saved error summary to: {ERROR_SUMMARY_PATH}")
    print(f"Saved misclassification pairs to: {MISCLASSIFICATION_PAIRS_PATH}")
    print(f"Saved permutation importance to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved hyperparameter stability table to: {HYPERPARAMETER_STABILITY_PATH}")


if __name__ == "__main__":
    main()
