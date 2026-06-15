# Assignment 4 - Reproducibility and Code Organization

**Project topic:** Music Genre Recognition Using Machine Learning

---

## A. Repository Structure and Modularity

The project retains the repository structure already used in the previous assignments:

```text
data/                    extracted feature tables
docs/                    assignment documents and generated experiment results
PDF/                     exploratory data analysis reports
src/                     feature extraction, modeling, and analysis scripts
experiment_config.json   shared experiment configuration
requirements.txt         pinned Python dependencies
README.md                project documentation
```

The recommended structure from the assignment was treated as guidance rather than a requirement to create empty directories. Separate `models/`, `notebooks/`, and `reports/` directories were not introduced because the current project does not use notebooks, does not require distribution of a serialized production model, and already stores generated CSV outputs consistently in `docs/`. This avoids restructuring completed work without an engineering need.

The source code is divided according to pipeline responsibility:

| Module | Responsibility |
|---|---|
| `src/project_config.py` | Shared configuration loading, path handling, data validation, and experiment manifest generation |
| `src/extract_features.py` | Extraction of 61 numerical features from GTZAN audio files |
| `src/baseline_models.py` | Training and evaluation of dummy, linear, tree, and ensemble baselines |
| `src/target_model.py` | RBF SVM optimization, feature-selection variant, and comparative evaluation |
| `src/results_analysis.py` | Consolidated results, class-level errors, permutation importance, and stability analysis |

The older `train.py` and `train_classifier.py` scripts are treated as historical exploratory scripts and are excluded from the final reproducibility workflow. This distinction is stated explicitly in the README.

Repository paths are resolved relative to the project root. The raw audio path is no longer embedded in the implementation. It is supplied through the `AUDIO_DATASET_PATH` environment variable. The canonical feature table remains `data/features_extended.csv`, and generated experimental outputs retain their established names in `docs/`.

This structure follows Assignment 3 directly:

```text
audio data
-> feature extraction
-> data validation
-> stratified split
-> model-specific pipeline
-> cross-validation
-> held-out evaluation
-> comparison and analysis
```

## B. Environment and Dependency Management

The reference technical environment is:

| Component | Specification |
|---|---|
| Operating system | Windows 10 Home, 64-bit |
| CPU | Intel Core i5-9400F, 6 cores at 2.90 GHz |
| RAM | Approximately 16 GB |
| GPU | Not required |
| Python | 3.14.0 |

Direct Python dependencies are pinned in `requirements.txt`:

| Library | Version |
|---|---:|
| `librosa` | 0.11.0 |
| `numpy` | 2.3.5 |
| `pandas` | 2.3.3 |
| `scikit-learn` | 1.8.0 |
| `matplotlib` | 3.10.8 |
| `seaborn` | 0.13.2 |
The environment is recreated with a local virtual environment and `python -m pip install -r requirements.txt`. The required Python version is stated in both `experiment_config.json` and the README.

The shared random seed is `30`. It controls the stratified train-test split, shuffled cross-validation folds, stochastic classifiers, and permutation-importance procedure. All main experiments use the same `20%` test split and five-fold stratified cross-validation.

Exact bit-level determinism can depend on the operating system, numerical-library build, and parallel execution with `n_jobs=-1`. These limitations do not change the defined data split or expected reported metrics, but the environment details are recorded in the experiment manifest.

The external resource dependency is the GTZAN audio collection. Raw audio is required only to regenerate the feature table. A reader who already has the versioned `data/features_extended.csv` can reproduce model training without access to the raw recordings.

## C. Version Control and Experimental Traceability

The project is maintained with Git. The existing history separates major research stages, including exploratory analysis, preprocessing correction, baseline implementation, target-model optimization, results analysis, methodology preparation, and literature review. Future commits should continue to identify the changed experimental stage rather than combine unrelated modifications.

Experiment tracking uses structured result files rather than an external tracking service. Every experiment writes CSV outputs to `docs/`, where the reported values can be inspected directly and connected to the assignment documents.

Each main script also appends a record to `docs/experiment_manifest.json`. The manifest contains:

- execution stage and UTC timestamp;
- current Git commit and dirty-worktree status;
- SHA-256 hash of the configuration;
- SHA-256 hash of the canonical feature table;
- hashes of the active source files;
- Python, operating-system, and main library versions.

The configuration hash identifies the parameter set, the dataset hash identifies the experimental input, and the source hashes identify the implementation used during execution. This also preserves traceability when an experiment is run before changes are committed.

Results intended for the thesis should be regenerated from a reviewed configuration and committed together with the corresponding CSV files and manifest. Git tags are optional and may be used for an approved thesis milestone, but meaningful commits remain the primary traceability mechanism.

## D. Configuration and Experiment Management

All controlled experimental parameters are stored in the single root file `experiment_config.json`. It defines:

- Python version;
- random seed, test size, cross-validation folds, and number of parallel jobs;
- primary optimization metric and target column;
- external audio-path environment variable;
- canonical feature-table path and output directory;
- baseline hyperparameter grids;
- RBF SVM and feature-selection search spaces;
- final analysis parameters.

The implementation logic remains in Python, while experiment values are loaded from JSON. The baseline, target-model, analysis, and extraction scripts accept:

```powershell
--config experiment_config.json
```

To create a controlled variant, the researcher copies the configuration, changes only the intended parameters, and passes the new file through `--config`. No manual modification of model parameters inside the scripts is required.

The shared data loader applies the same duplicate-removal and missing-value checks in every experiment. Scaling and feature selection remain inside scikit-learn pipelines, so they are fitted only on the corresponding training data. This preserves the leakage controls and controlled comparison defined in Assignment 3.

The configuration does not change the established methodology. It formalizes the values previously present as constants in several scripts. The same seed, split, cross-validation, model grids, metrics, input data, and output filenames are retained.

## E. Documentation and README Standards

One root `README.md` is used as the complete technical documentation. No additional README files are introduced. It contains:

- the project objective and research context;
- the existing repository structure;
- module responsibilities;
- hardware, Python, and dependency requirements;
- installation instructions;
- configuration rules;
- raw-data setup and preprocessing instructions;
- execution commands for baseline training, target-model optimization, and analysis;
- pipeline stages;
- expected output files;
- the reproducibility protocol.

A technically competent reader can install the project with:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The reader can then reproduce the completed modeling workflow with:

```powershell
python src\baseline_models.py --config experiment_config.json
python src\target_model.py --config experiment_config.json
python src\results_analysis.py --config experiment_config.json
```

Feature extraction is run separately because it requires the external audio collection:

```powershell
$env:AUDIO_DATASET_PATH = "D:\path\to\genres_original"
python src\extract_features.py --config experiment_config.json
```

The extraction script writes through a temporary file and refuses to replace the existing feature table when no audio features are extracted. This prevents an invalid source path from silently destroying the canonical dataset.

The reproducibility procedure is:

1. check out the code state associated with the experiment;
2. recreate the documented environment;
3. use the configuration identified by its hash;
4. verify the feature-table hash;
5. execute the pipeline stages in order;
6. compare the generated CSV outputs with the reported results.

The resulting organization satisfies Assignment 4 without changing the completed repository layout or introducing directories that the project does not use.
