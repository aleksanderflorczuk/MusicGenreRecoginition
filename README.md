# Music Genre Recognition Using Machine Learning

## Project Objective

The project investigates whether ten music genres can be recognized from 61
engineered audio features and whether an RBF support vector machine improves
over simpler machine-learning baselines under the controlled protocol defined
in Assignment 3.

## Existing Repository Structure

```text
data/                    Extracted feature tables
docs/                    Assignment documents and generated CSV results
PDF/                     Exploratory data analysis reports
src/                     Feature extraction, modeling, and analysis scripts
experiment_config.json   Shared experiment configuration
requirements.txt         Pinned Python dependencies
README.md                Project documentation
```

The current structure is intentionally retained. The source code is stored in
`src/`, the canonical unscaled dataset is stored in `data/`, and all generated
experimental results remain in `docs/`.

The reproducible workflow uses:

- `src/project_config.py` for configuration, paths, validated data loading, and
  experiment manifests;
- `src/extract_features.py` for audio feature extraction;
- `src/baseline_models.py` for baseline comparison;
- `src/target_model.py` for RBF SVM optimization;
- `src/results_analysis.py` for final analytical tables.

The older `train.py` and `train_classifier.py` scripts are exploratory and are
not part of the final reproducibility protocol.

## Environment

Reference environment:

| Component | Specification |
|---|---|
| Operating system | Windows 10 Home, 64-bit |
| CPU | Intel Core i5-9400F, 6 cores |
| RAM | Approximately 16 GB |
| GPU | Not required |
| Python | 3.14.0 |

Create and activate the environment from the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Paths, random seed, split settings, cross-validation settings, model grids,
and analysis parameters are stored in `experiment_config.json`. Parameters
should be changed in a copied configuration file rather than directly inside
the Python scripts.

Every main script accepts:

```powershell
--config experiment_config.json
```

The default experiment uses:

- random seed `30`;
- stratified test size `0.2`;
- five stratified cross-validation folds;
- accuracy as the optimization metric;
- `data/features_extended.csv` as the canonical modeling table;
- `docs/` as the output directory.

## Data Preparation

Raw GTZAN audio is an external dependency and is not included in the
repository. To regenerate the feature table, set its location for the current
PowerShell session:

```powershell
$env:AUDIO_DATASET_PATH = "D:\path\to\genres_original"
python src\extract_features.py --config experiment_config.json
```

The script extracts tempo, zero-crossing rate, RMS energy, spectral
descriptors, MFCC statistics, and chroma statistics. It writes the unscaled
feature table to the path defined in the configuration.

If `data/features_extended.csv` is already present and the raw audio has not
changed, feature extraction does not need to be repeated.

## Pipeline Execution

Run the stages from the project root in this order:

```powershell
python src\baseline_models.py --config experiment_config.json
python src\target_model.py --config experiment_config.json
python src\results_analysis.py --config experiment_config.json
```

The pipeline is:

```text
GTZAN audio
-> feature extraction
-> duplicate and missing-value validation
-> fixed stratified train/test split
-> preprocessing inside model pipelines
-> baseline evaluation
-> RBF SVM optimization
-> held-out evaluation
-> result and error analysis
```

## Expected Outputs

The scripts update the existing result files in `docs/`:

- `assignment7_baseline_results.csv`;
- `assignment7_classification_reports.csv`;
- `assignment7_best_confusion_matrix.csv`;
- `assignment8_target_results.csv`;
- `assignment8_comparative_results.csv`;
- `assignment8_optimization_results.csv`;
- `assignment8_learning_curve.csv`;
- `assignment8_classification_report.csv`;
- `assignment8_best_confusion_matrix.csv`;
- `assignment9_consolidated_results.csv`;
- `assignment9_error_summary.csv`;
- `assignment9_misclassification_pairs.csv`;
- `assignment9_permutation_importance.csv`;
- `assignment9_hyperparameter_stability.csv`.

Each stage also appends execution metadata to
`docs/experiment_manifest.json`. The manifest records the Git commit,
worktree state, configuration hash, dataset hash, Python version, operating
system, core library versions, and hashes of the active source files.

## Reproducibility Protocol

1. Check out the Git commit associated with the reported experiment.
2. Create Python 3.14.0 environment and install `requirements.txt`.
3. Use the configuration identified by the manifest hash.
4. Confirm that the feature-table hash matches the manifest.
5. Run the three pipeline scripts in the documented order.
6. Compare the generated CSV files with the thesis results.

Major experimental changes should be committed separately with messages that
identify the changed data, preprocessing, model, or evaluation setting. A Git
tag may be added for an approved thesis milestone.
