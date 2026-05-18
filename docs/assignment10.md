# Assignment 10 - Methodology and Technical Implementation Chapters

## 1. Objective and Position in the Thesis

This document transforms the completed experimental work from Assignments 1-9 into a formal draft of the practical thesis chapters. The purpose is not to describe the work chronologically, but to present the methodological and technical logic of the project: the research problem defines the data requirements, the data determine the representation, the representation constrains the model choice, the validation protocol controls the comparison, and the results define the limitations of the final system.

The project investigates automatic music genre recognition from extracted audio features. The practical work uses a benchmark-style ten-class music genre dataset and represents each audio recording as a fixed-length numerical feature vector. The final experimental evidence comes from the controlled comparison between baseline machine learning models and the primary support vector machine model developed in the later stages of the project.

The methodological foundation is based on the exploratory and preprocessing conclusions documented in `PDF/eda_s27833_2026v2.pdf`, `docs/assignment6.md`, and the experimental outputs from Assignments 7-9. The main implementation artifacts are located in `src/extract_features.py`, `src/eda.py`, `src/baseline_models.py`, `src/target_model.py`, and `src/results_analysis.py`.

## 2. Draft Methodology Chapter

### A. Methodological Framework

#### Research Problem

The research problem addressed in this thesis is the automatic recognition of music genre from audio-derived features. In practical terms, the task is a supervised multi-class classification problem: each input example is an audio track represented by numerical descriptors, and the target output is one of ten genre labels. The genre classes are `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`.

The project is motivated by the assumption that genre membership is reflected in measurable acoustic properties of music recordings. These properties include timbre, energy, spectral distribution, rhythmic speed, and harmonic content. The research question can therefore be stated as follows:

```text
Can music genre be recognized automatically from extracted audio features using classical machine learning methods?
```

The operational hypothesis used in the experimental phase is that a machine learning classifier trained on extracted audio descriptors will achieve at least 70% classification accuracy on a benchmark-style ten-class music genre dataset. This threshold is intentionally higher than the expected chance-level performance of approximately 10% in a balanced ten-class problem, but it remains realistic for a system based on global engineered features rather than raw audio or deep spectrogram models.

#### Experimental Design

The experiment was designed as a controlled comparison between several model families trained on the same cleaned feature table. The design follows a fixed sequence of methodological decisions:

1. Audio recordings are converted into a tabular representation using reproducible feature extraction.
2. The extracted table is cleaned by removing exact duplicate rows and checking missing values.
3. A stratified train-test split is created to preserve the class distribution.
4. Preprocessing is fitted only inside training folds or the training split.
5. Baseline models are evaluated to establish a reference level of performance.
6. The primary model is optimized under the same validation protocol.
7. The final comparison is made using cross-validation results, held-out test metrics, class-level errors, and robustness indicators.

This design separates exploratory analysis from final model evaluation. Exploratory scaling and PCA visualizations were useful for understanding the feature space, but final modeling used `data/features_extended.csv`, not the globally scaled exploratory file. This distinction is important because fitting transformations before train-test splitting would allow information from the test distribution to influence the training procedure.

#### What Is Being Tested

The experiment tests whether a fixed set of engineered audio descriptors contains enough information to classify music genre above a predefined performance threshold. It also tests whether a non-linear margin-based classifier can improve over classical baseline models under the same data split, preprocessing rules, and evaluation metrics.

The primary model is an RBF-kernel support vector machine. Its role is not simply to add complexity, but to test a specific modeling assumption: genre separation in the engineered feature space may require non-linear boundaries, but the dataset size does not justify a high-capacity neural architecture. The RBF SVM therefore represents a controlled increase in model flexibility over linear and tree-based baselines.

#### Conditions of the Experiment

All models were evaluated under the same core conditions:

| Element | Configuration |
|---|---|
| Input file | `data/features_extended.csv` |
| Initial observations | 1000 |
| Removed duplicates | 13 exact duplicate rows |
| Final observations | 987 |
| Predictors | 61 numerical audio features |
| Target | `genre` |
| Test split | 20% |
| Split type | Stratified train-test split |
| Random seed | 30 |
| Cross-validation | 5-fold stratified CV on the training set |
| Primary metric | Accuracy |
| Supporting metrics | Macro precision, macro recall, macro F1 |
| Leakage control | Preprocessing fitted inside pipelines only |

The test set was not used for hyperparameter selection. Hyperparameters were selected using cross-validation on the training data, and the held-out test set was used only for final evaluation. This rule is central to the validity of the reported performance.

#### Evaluation Strategy

Accuracy was selected as the primary metric because the dataset is approximately balanced across the ten classes. In this context, accuracy has a clear interpretation: it measures the proportion of correctly classified tracks. However, accuracy alone is insufficient because genre-level weaknesses may be hidden by a good aggregate score. Therefore, macro precision, macro recall, and macro F1 were used as supporting metrics.

Macro F1 is especially relevant because it gives equal importance to each genre, regardless of small differences in class support after duplicate removal and splitting. Confusion matrices and class-level summaries were used to determine which genres were systematically difficult. This is necessary for thesis-level interpretation because a model with acceptable aggregate accuracy may still fail on musically important subsets of the task.

The final evaluation also includes robustness indicators:

- cross-validation standard deviation;
- train-validation generalization gap;
- comparison between cross-validation and held-out test accuracy;
- learning-curve behavior;
- hyperparameter stability near the selected configuration;
- permutation importance for model-agnostic feature interpretation.

#### Comparison Logic Between Models

The comparison logic is based on methodological fairness. A model is considered better only if it improves over the baseline under the same data, split, preprocessing constraints, and evaluation metrics. This prevents an invalid comparison in which one model benefits from a different split, a different feature table, or preprocessing fitted on the full dataset.

The baseline stage included a naive reference model, a linear model, a single tree, and ensemble models:

| Model | Methodological role |
|---|---|
| Dummy classifier | Establishes chance-level reference for a balanced ten-class task. |
| Logistic regression | Tests whether the standardized feature space is sufficiently linearly separable. |
| Decision tree | Tests a simple non-linear threshold-based model. |
| Random forest | Provides a robust classical tabular baseline using many trees. |
| Gradient boosting | Tests a sequential ensemble baseline for structured features. |

The strongest baseline was the default random forest, with held-out test accuracy of 0.6919 and macro F1 of 0.6892. The optimized RBF SVM achieved test accuracy of 0.7121 and macro F1 of 0.7121. The absolute improvement over the strongest baseline was approximately 0.0202 accuracy and 0.0228 macro F1. This is a meaningful but moderate improvement. Therefore, the final claim should be that the hypothesis is supported under the current protocol, not that the genre recognition problem is solved.

### B. Dataset Description

#### Dataset Characteristics

The dataset consists of audio recordings grouped into ten music genres. The extracted feature table contains one row per track and one target label. Before cleaning, the table contained 1000 observations and 62 columns: 61 numerical predictors and one categorical target variable. Each genre initially had 100 recordings, which makes the dataset suitable for accuracy-based evaluation because no single class dominates the label distribution.

After exact duplicate removal, the final experimental dataset contained 987 observations. The removal of 13 duplicate rows was methodologically important because duplicate observations can inflate evaluation results if identical or near-identical examples appear in both training and test subsets. The cleaning step therefore improves the credibility of the held-out test evaluation.

The class distribution remains close to balanced after cleaning, although it is no longer perfectly equal because duplicate removal affected some genres more than others:

| Class | Observations after cleaning |
|---|---:|
| blues | 100 |
| classical | 100 |
| country | 100 |
| disco | 99 |
| hiphop | 98 |
| jazz | 100 |
| metal | 93 |
| pop | 98 |
| reggae | 99 |
| rock | 100 |

The final feature matrix contains only numerical predictors. The target variable is the genre label. No missing values were detected, so imputation was not required. This simplifies the preprocessing design and reduces the number of assumptions introduced before modeling.

#### Analytical Interpretation of the Dataset

The dataset is suitable for a controlled engineering experiment because it is small enough to support reproducible model comparison and large enough to evaluate several classical machine learning methods. The balanced class structure makes the experimental objective clear: a trivial classifier is expected to perform near 10% accuracy, while learned models must demonstrate a substantial improvement over this lower bound.

At the same time, the dataset imposes important limitations. A final size of 987 observations is relatively small for a ten-class audio classification task. The feature representation compresses each track into global statistics rather than preserving the full waveform or time-frequency sequence. As a result, the dataset supports a controlled classical machine learning study, but it does not support strong claims about general-purpose music understanding.

#### Data Splits and Validation Strategy

The data were split using a stratified 80/20 train-test procedure with random seed 30. Stratification preserves the class distribution in both subsets and is important because each class should be represented in the final held-out evaluation.

The training split was used for model fitting and hyperparameter selection through 5-fold stratified cross-validation. The test split was reserved for final evaluation. This structure creates a clear separation between model selection and model assessment:

| Stage | Role |
|---|---|
| Training folds | Fit models and preprocessing transformations. |
| Validation folds | Select hyperparameters and estimate validation stability. |
| Held-out test set | Estimate final generalization performance once per selected model. |

This validation strategy ensures that the reported final test results are not directly optimized during model selection.

#### Constraints and Limitations

The main dataset limitation is representational rather than purely statistical. The recordings are represented by global descriptors such as means and standard deviations. This representation captures broad timbral, spectral, energy, and harmonic characteristics, but it cannot directly encode temporal progression, groove, rhythmic patterns, or local arrangement changes. These missing temporal aspects are especially relevant for genres such as disco, reggae, country, and rock, where genre identity may depend strongly on rhythm, instrumentation patterns, and stylistic context.

The second limitation is dataset size. Although 987 examples are sufficient for classical model comparison, the number of observations per class remains modest. This increases the risk that a flexible model can fit training folds very strongly while still failing to generalize perfectly. The final RBF SVM shows this pattern: it has high training accuracy and a substantial train-validation gap, even though its validation and test accuracy remain reasonably close.

### C. Preprocessing and Feature Engineering

#### Data Cleaning Strategy

The preprocessing strategy begins with validation of the extracted feature table. The table was checked for missing values, duplicate rows, target labels, and numeric predictor structure. Since no missing values were found, no imputation was introduced. This is preferable to adding an unnecessary imputation step, because each additional preprocessing stage creates assumptions that must be justified and reproduced.

Exact duplicate rows were removed before splitting. This decision protects the independence of the held-out test set. If duplicates were left in the table, the same feature vector could appear in both training and test data, making the final accuracy less informative. Removing duplicates therefore supports a more conservative estimate of generalization.

#### Feature Representation

The central feature engineering decision is the conversion of each raw audio file into a fixed-length numerical vector. The extracted feature groups are:

| Feature group | Description | Methodological role |
|---|---|---|
| Tempo | Estimated rhythmic speed in beats per minute. | Captures broad rhythmic speed differences. |
| Zero crossing rate | Mean and standard deviation. | Describes noisiness and high-frequency signal behavior. |
| RMS energy | Mean and standard deviation. | Represents loudness and energy dynamics. |
| Spectral centroid | Mean and standard deviation. | Captures perceived brightness of the sound. |
| Spectral bandwidth | Mean and standard deviation. | Represents spectral spread. |
| Spectral rolloff | Mean and standard deviation. | Captures upper-frequency energy distribution. |
| MFCC | 13 means and 13 standard deviations. | Represents timbral characteristics. |
| Chroma | 12 means and 12 standard deviations. | Represents pitch-class and harmonic distribution. |

This representation was selected because it is compatible with classical machine learning algorithms and is reproducible from raw `.wav` files using deterministic feature extraction. It also gives each model access to complementary musical information: timbre, energy, spectrum, tempo, and harmony.

The choice to summarize time-varying descriptors using means and standard deviations is a deliberate engineering trade-off. It reduces variable-length audio signals to a uniform tabular format, which makes baseline comparison straightforward. However, it also removes sequential information. Therefore, the representation is expected to work better for genres with distinctive global timbral or spectral profiles than for genres distinguished mainly by temporal rhythm or arrangement.

#### Scaling and Leakage Control

Standardization was used for models sensitive to feature scale. The project contains descriptors measured on different numerical scales: tempo is expressed in beats per minute, spectral descriptors use frequency-related values, and MFCC and chroma coefficients are unitless. Without scaling, distance- and margin-based models would be dominated by features with larger numerical ranges.

Scaling was implemented inside scikit-learn pipelines. This means that the scaler is fitted only on the training data in each fold and then applied to the corresponding validation or test data. This is essential for leakage control. The globally scaled file `data/features_extended_scaled.csv` was treated only as an exploratory artifact and was not used as the final modeling input.

The methodological rule is:

```text
split first, fit preprocessing second, evaluate third
```

This rule applies to standardization, feature selection, and any other transformation whose parameters are learned from data.

#### Encoding Procedures

No categorical predictors were present in the feature table. Therefore, one-hot encoding, ordinal encoding, and target encoding were not required. The only categorical variable was the target label `genre`. The models used in the project can operate with string class labels, so no additional target transformation was necessary for the reported experiments.

#### Feature Selection

Feature selection was evaluated as an experimental variant of the primary model using `SelectKBest` with the ANOVA F-score. This variant tested whether removing potentially noisy or redundant predictors could improve generalization. The selector was placed inside the pipeline so that feature selection was fitted only within training folds.

The result did not support feature removal. The best feature-selection configuration selected `k="all"`, producing the same validation and test performance as the all-feature RBF SVM. This negative result is methodologically useful. It shows that the final model should not include an additional feature-selection step merely for complexity. The simpler all-feature SVM is preferred because it achieves the same result with a cleaner pipeline and fewer moving parts.

### D. Model Architecture and Configuration

#### Baseline Models

The baseline models were selected to provide an empirical performance ladder. The dummy classifier establishes the lower bound, logistic regression tests linear separability, decision tree tests a simple non-linear model, and ensemble models test stronger classical tabular approaches.

| Model | Configuration logic | Interpretation |
|---|---|---|
| Dummy classifier | Most frequent class strategy. | Confirms that learned models outperform a naive reference. |
| Logistic regression | Standardized features, limited `C` search. | Tests whether genre separation is mostly linear. |
| Decision tree | Default and limited tuned variants. | Tests simple non-linear splits but exposes overfitting risk. |
| Random forest | Default and limited tuned variants. | Provides robust non-linear baseline for engineered tabular features. |
| Gradient boosting | Default and limited tuned variants. | Tests sequential ensemble learning on the same features. |

The best baseline was the default random forest. It achieved 0.6919 held-out test accuracy and 0.6892 macro F1. This result demonstrates that the extracted features contain meaningful genre information, but it falls slightly below the 70% hypothesis threshold.

#### Primary Model

The primary model is a support vector machine with a radial basis function kernel:

```text
StandardScaler -> SVC(kernel="rbf")
```

The RBF SVM was selected because the baseline results suggested that the classification boundary is not purely linear. Logistic regression reached only 0.6111 test accuracy, while the random forest performed substantially better. This pattern indicates that non-linear feature interactions are relevant. The RBF SVM provides a controlled way to model non-linear boundaries in a standardized feature space without moving to a neural model that would be difficult to justify for a dataset of fewer than 1000 cleaned examples.

The optimized final configuration was:

```text
SVC(kernel="rbf", C=3, gamma="scale")
```

The hyperparameter `C` controls the trade-off between margin width and training errors. The selected value `C=3` represents moderate regularization. The `gamma` parameter controls the locality of the RBF kernel. The selected value `gamma="scale"` adapts the kernel width to the feature variance after scaling.

#### Optimization Strategy

Hyperparameter optimization was performed with grid search and 5-fold stratified cross-validation on the training data. The search space was intentionally bounded:

| Component | Search values |
|---|---|
| `C` | `[1, 3, 10, 30, 100]` |
| `gamma` | `["scale", 0.03, 0.01, 0.003, 0.001]` |
| `selector__k` in optional variant | `[20, 30, 40, 50, "all"]` |

The purpose of this search was not to exhaustively maximize the test score, but to evaluate a plausible range of regularization and kernel-width settings while preserving methodological control. Larger values of `C` were included to test whether increased model capacity improves validation performance. The results showed that higher capacity increased training accuracy but did not improve validation accuracy.

#### Evaluation Fairness

All baseline and primary models used the same cleaned dataset, split strategy, random seed, cross-validation design, and final test set. This allows the final comparison to be interpreted as a model effect rather than a consequence of different data preparation.

The final consolidated comparison is:

| Model | Role | CV accuracy mean | CV accuracy std | Test accuracy | Test macro F1 |
|---|---|---:|---:|---:|---:|
| SVM RBF all features | Primary model | 0.7275 | 0.0130 | 0.7121 | 0.7121 |
| SVM RBF SelectKBest | Optional variant | 0.7275 | 0.0130 | 0.7121 | 0.7121 |
| Random forest default | Strongest baseline | 0.6692 | 0.0309 | 0.6919 | 0.6892 |
| Random forest tuned | Tuned baseline | 0.6806 | 0.0166 | 0.6818 | 0.6795 |
| Gradient boosting tuned | Tuned baseline | 0.6388 | 0.0129 | 0.6465 | 0.6470 |
| Logistic regression default | Linear baseline | 0.6831 | 0.0339 | 0.6111 | 0.6061 |
| Decision tree default | Simple non-linear baseline | 0.4309 | 0.0214 | 0.4495 | 0.4464 |
| Dummy classifier | Naive reference | 0.1014 | 0.0003 | 0.1010 | 0.0183 |

The SVM improves over the strongest baseline by approximately 2.02 percentage points in accuracy and 2.28 percentage points in macro F1. The improvement is consistent across both primary metrics, which supports the selection of the RBF SVM as the final model. However, the improvement is moderate and should be interpreted with restraint.

#### Results and Model Behavior

The final SVM achieved:

| Metric | Value |
|---|---:|
| Mean CV accuracy | 0.7275 |
| CV accuracy standard deviation | 0.0130 |
| Mean training accuracy in CV | 0.9861 |
| Train-validation gap | 0.2585 |
| Test accuracy | 0.7121 |
| Test macro precision | 0.7161 |
| Test macro recall | 0.7122 |
| Test macro F1 | 0.7121 |

The held-out test accuracy exceeds the predefined 70% threshold, so the project hypothesis is supported under the defined experimental conditions. The small difference between cross-validation accuracy and test accuracy suggests that the final test result is not a severe collapse relative to validation performance.

At the same time, the train-validation gap is substantial. The model fits the training folds very strongly, which is expected for a flexible RBF kernel. This does not invalidate the result, but it limits the strength of the conclusion. The final claim should therefore emphasize controlled improvement and empirical support, not complete generalization.

#### Error Patterns and Limitations

The final model does not perform equally well across genres. The weakest classes are `disco` and `rock`. Disco has recall and precision of 0.45, while rock has recall of 0.50 and precision of approximately 0.5263. The strongest class is `classical`, with recall of 0.95 and precision of approximately 0.9048.

The main class-level pattern is:

| Class | Support | Correct | Incorrect | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| disco | 20 | 9 | 11 | 0.4500 | 0.4500 |
| rock | 20 | 10 | 10 | 0.5000 | 0.5263 |
| country | 20 | 13 | 7 | 0.6500 | 0.5909 |
| hiphop | 20 | 13 | 7 | 0.6500 | 0.8667 |
| reggae | 20 | 14 | 6 | 0.7000 | 0.6364 |
| metal | 18 | 13 | 5 | 0.7222 | 0.7647 |
| blues | 20 | 16 | 4 | 0.8000 | 0.7619 |
| pop | 20 | 17 | 3 | 0.8500 | 0.8095 |
| jazz | 20 | 17 | 3 | 0.8500 | 0.8500 |
| classical | 20 | 19 | 1 | 0.9500 | 0.9048 |

The most frequent specific confusion is `disco` predicted as `country`, occurring four times. Other notable confusions include `hiphop` predicted as `reggae` and `metal` predicted as `disco`. These errors are musically plausible because several genres share instrumentation, energy patterns, and production characteristics.

The error analysis supports the central limitation of the representation. Global summary features capture average timbre, energy, and spectral shape, but they do not directly encode rhythmic pattern, groove, or temporal arrangement. This explains why genres with strong temporal or stylistic overlap remain difficult even when the aggregate model score is acceptable.

### E. Experimental Environment

The experiments were conducted in a local Python environment. The environment details are important because reproducibility requires both code-level and dependency-level transparency.

| Component | Specification |
|---|---|
| Operating system | Microsoft Windows 10 Home, 64-bit |
| CPU | Intel Core i5-9400F, 6 cores / 6 logical processors |
| RAM | Approximately 16 GB |
| Python | 3.14.0 |
| Main audio library | `librosa 0.11.0` |
| Machine learning library | `scikit-learn 1.8.0` |
| Data processing | `pandas 2.3.3`, `numpy 2.3.5` |
| Visualization | `matplotlib 3.10.8`, `seaborn 0.13.2` |
| Scientific stack | `scipy 1.16.3`, `joblib 1.5.3` |

The code fixes random seeds where stochastic behavior affects splitting or model training. The main random seed is 30. Reproducibility is also supported by saving intermediate and final outputs as CSV files in the `docs` directory.

### F. Writing Standards and Academic Tone

The methodology is written as an explanation of design logic rather than a chronological account of implementation. The text avoids informal statements such as "I tested several models" and instead describes the experiment as a structured process. The intended thesis style is:

```text
The models were evaluated under a fixed stratified split and cross-validation protocol.
```

instead of:

```text
I trained the models and checked which one worked best.
```

Terminology should remain consistent throughout the thesis. The following terms should be used with stable meanings:

| Term | Meaning in this thesis |
|---|---|
| Feature extraction | Conversion from raw audio files into numerical descriptors. |
| Preprocessing | Data cleaning, scaling, and pipeline-controlled transformations. |
| Baseline model | Reference model used before the primary model. |
| Primary model | Final optimized RBF SVM. |
| Validation | Cross-validation on the training set. |
| Test evaluation | Final held-out evaluation after model selection. |
| Leakage | Any use of validation or test information during fitting of preprocessing or model parameters. |

The final thesis chapter should maintain a balance between technical specificity and readability. It should include enough detail to reproduce the experiment, but it should not include long code listings. Code fragments should be used only when they clarify the pipeline structure or a methodological rule.

## 3. Draft Technical Implementation Chapter

### 3.1 System Overview

The technical implementation consists of a reproducible pipeline for converting audio files into a feature table, evaluating baseline models, optimizing the primary model, and generating interpretation artifacts. The implementation is organized into separate scripts so that each stage has a clear responsibility.

| Script | Responsibility |
|---|---|
| `src/extract_features.py` | Extracts numerical audio descriptors from `.wav` files and saves `features_extended.csv`. |
| `src/eda.py` | Performs exploratory analysis, distribution plots, correlation analysis, and PCA visualization. |
| `src/baseline_models.py` | Trains and evaluates baseline classifiers under the fixed protocol. |
| `src/target_model.py` | Optimizes and evaluates the primary RBF SVM model and its feature-selection variant. |
| `src/results_analysis.py` | Consolidates results and produces error, importance, and stability analyses. |

The implementation follows the same conceptual order as the methodology: data are first represented as features, then cleaned and split, then evaluated under controlled validation, and finally interpreted through aggregate and class-level results.

### 3.2 Feature Extraction Implementation

Feature extraction is implemented in `src/extract_features.py`. The script iterates through genre folders, loads `.wav` files with `librosa`, extracts numerical descriptors, appends the genre label, and writes the resulting table to `data/features_extended.csv`.

The raw audio path is defined in `config.py`:

```text
AUDIO_DATASET_PATH = "D:/genre music dataset/genres_original"
```

The extractor preserves the original sampling rate by loading files with `sr=None`. This avoids forcing all recordings into a fixed sampling rate during loading. Each recording is transformed into one row with 61 predictors. The target label is derived from the folder name, which keeps label assignment consistent with the dataset structure.

The technical feature extraction process has two important methodological consequences. First, it produces a fixed-size tabular representation, which allows classical machine learning models to be used. Second, it compresses each track into global summary values, which makes the system computationally simple but limits its ability to model temporal structure.

### 3.3 Exploratory Data Analysis Implementation

The exploratory stage is implemented in `src/eda.py` and documented in `PDF/eda_s27833_2026v2.pdf`. The EDA checks the shape of the dataset, missing values, class distribution, feature distributions, correlations, and PCA visualization of selected feature groups.

The EDA supported several later methodological decisions:

- the dataset was balanced enough to use accuracy as the primary metric;
- missing values were absent, so imputation was not required;
- feature scales differed substantially, so scaling was required for SVM and logistic regression;
- global spectral features showed correlation, so interpretability required caution;
- PCA suggested partial genre separability but also overlap between classes.

The EDA did not directly determine final model performance. Its role was to identify preprocessing risks and guide the experimental design.

### 3.4 Baseline Model Implementation

Baseline evaluation is implemented in `src/baseline_models.py`. The script loads `data/features_extended.csv`, removes exact duplicate rows, checks for missing values, creates a stratified train-test split, and evaluates default and tuned baseline models.

The baseline script saves:

| Output file | Content |
|---|---|
| `docs/assignment7_baseline_results.csv` | Aggregate baseline metrics and best parameters. |
| `docs/assignment7_classification_reports.csv` | Class-level reports for baseline models. |
| `docs/assignment7_best_confusion_matrix.csv` | Confusion matrix for the best baseline model. |

Each model is evaluated in a scikit-learn pipeline. This implementation choice is important because the same fitted preprocessing object travels with the estimator. It prevents accidental use of full-dataset scaling and ensures that cross-validation estimates include preprocessing variance.

The baseline results identified the default random forest as the strongest reference model. This result established the performance threshold that the primary model needed to exceed under the same protocol.

### 3.5 Primary Model Implementation

The primary model is implemented in `src/target_model.py`. The script evaluates two related experiments:

```text
StandardScaler -> SVC(kernel="rbf")
```

and:

```text
StandardScaler -> SelectKBest(f_classif) -> SVC(kernel="rbf")
```

The first experiment uses all 61 features. The second experiment tests whether a filter-based feature-selection step improves generalization. Both experiments are optimized with `GridSearchCV` using 5-fold stratified cross-validation on the training split.

The script saves:

| Output file | Content |
|---|---|
| `docs/assignment8_target_results.csv` | Final metrics for SVM experiments. |
| `docs/assignment8_comparative_results.csv` | Comparison between best baseline and primary model. |
| `docs/assignment8_optimization_results.csv` | Full grid-search results. |
| `docs/assignment8_learning_curve.csv` | Learning-curve results for the selected model. |
| `docs/assignment8_classification_report.csv` | Class-level report for the selected model. |
| `docs/assignment8_best_confusion_matrix.csv` | Confusion matrix for the selected model. |

The selected model was the all-feature RBF SVM with `C=3` and `gamma="scale"`. The feature-selection variant selected `k="all"` and achieved the same result, so the additional selector was not retained as part of the final system.

### 3.6 Results Analysis Implementation

The final interpretation artifacts are generated by `src/results_analysis.py`. This script does not redesign the experiment. It reuses the same cleaned data, stratified split, and final SVM configuration to compute additional analysis tables.

The generated outputs are:

| Output file | Content |
|---|---|
| `docs/assignment9_consolidated_results.csv` | Unified ranking of baseline and primary models. |
| `docs/assignment9_error_summary.csv` | Class-level correct and incorrect predictions. |
| `docs/assignment9_misclassification_pairs.csv` | Most frequent true-predicted class confusions. |
| `docs/assignment9_permutation_importance.csv` | Model-agnostic feature importance estimates. |
| `docs/assignment9_hyperparameter_stability.csv` | Top SVM configurations and generalization gaps. |

The permutation importance results show that the final model relies strongly on MFCC and RMS features. The top-ranked feature is `mfcc_9_mean`, followed by features such as `mfcc_5_std`, `rms_std`, `mfcc_3_mean`, and `mfcc_11_mean`. This is consistent with the expectation that timbre and energy are central to genre discrimination in a global-feature representation.

### 3.7 Reproducibility Controls

The implementation includes several reproducibility controls:

- the feature extraction definitions are fixed in `src/extract_features.py`;
- exact duplicate rows are removed before splitting;
- missing values cause an explicit error rather than silent model fitting;
- the split uses `random_state=30`;
- cross-validation uses stratified folds;
- preprocessing and feature selection are fitted only inside pipelines;
- final metrics and analysis tables are saved as CSV artifacts.

These controls make the experiment reproducible at the level required for an engineering thesis. A future researcher can inspect the input file, rerun the scripts, and compare the generated CSV outputs with the reported tables.

### 3.8 Technical Limitations

The implementation is intentionally classical and tabular. It does not process raw waveforms directly during model training and does not use spectrograms, convolutional networks, recurrent models, or transformer-based audio encoders. This is a methodological choice aligned with the project scope, but it limits the system's ability to learn temporal patterns.

The feature extraction script summarizes each track using global statistics. This design is efficient and reproducible, but it removes local structure. The error analysis shows that this limitation is not merely theoretical: disco, rock, country, and reggae remain difficult because they often differ through rhythm, arrangement, and stylistic context that are only indirectly represented by the current features.

The final model also shows overfitting indicators. Its mean training accuracy in cross-validation is 0.9861, while its mean validation accuracy is 0.7275. The held-out test score remains close to the validation score, but the large train-validation gap shows that the RBF SVM has high capacity relative to the dataset size. This supports a cautious conclusion: the system meets the predefined accuracy target, but further improvement would likely require richer representations, more data, or temporal feature engineering rather than only stronger tuning of the same model.

## 4. Integrated Chapter Conclusion

The methodology and technical implementation are coherent with the research problem. The project asks whether music genre can be recognized from extracted audio features. The dataset provides an initially balanced and, after duplicate removal, still approximately balanced ten-class classification setting. The feature representation captures timbral, energy, spectral, rhythmic-speed, and harmonic properties. The validation design prevents leakage through stratified splitting, cross-validation, and pipeline-based preprocessing. The model comparison shows that the final RBF SVM outperforms the strongest baseline and exceeds the predefined 70% accuracy threshold.

The final result supports the project hypothesis under the defined experimental conditions. The optimized SVM achieved 0.7121 test accuracy and 0.7121 macro F1, compared with 0.6919 test accuracy and 0.6892 macro F1 for the strongest baseline. The improvement is real but moderate. The thesis should therefore present the final system as a valid controlled engineering solution, not as a complete solution to music genre recognition.

The main limitation is the global feature representation. It is sufficient to separate some genres with distinctive timbral or spectral profiles, especially classical, jazz, pop, blues, and metal. It is less effective for genres whose identity depends on temporal rhythm, arrangement, and overlap with neighboring styles. This limitation should guide the later conclusion and future work chapters.
