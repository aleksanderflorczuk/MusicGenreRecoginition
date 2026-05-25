# Assignment 11 - Full Practical Part Review

This self-assessment identifies the main strengths of the practical component as methodological coherence, fair model comparison, reproducible evaluation, and cautious interpretation of results. The main weaknesses are the moderate improvement over the strongest baseline, the high train-validation gap of the final SVM, the limited dataset size, and the restricted expressive power of global audio features. The review below verifies whether the practical part is ready for final consolidation without introducing unnecessary new experiments.

## A. Logical Continuity Verification

The practical component is logically continuous with the research question. The thesis investigates whether music genre can be automatically recognized from extracted audio features using machine learning methods. This question is operationalized as a supervised ten-class classification task, where each music track is represented by numerical descriptors such as tempo, RMS energy, zero crossing rate, spectral features, MFCC statistics, and chroma features. The target variable is one of ten genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`.

The research question is directly addressed by the final results. The optimized RBF-kernel SVM achieved test accuracy of 0.7121 and macro F1 of 0.7121. This exceeds the predefined 70% accuracy threshold and therefore supports the operational hypothesis under the defined experimental conditions. The result should not be interpreted as proof that music genre recognition is solved in a general sense. It shows that automatic genre recognition is feasible for the current dataset, feature representation, and evaluation protocol.

The identified research gap is meaningfully engaged. The project evaluates whether global engineered audio descriptors are sufficient for genre recognition and where their limitations appear. The results show that these descriptors contain useful genre-discriminative information, because all learned models substantially outperform the dummy classifier. At the same time, the error analysis shows that global summary features are weaker for genres whose identity depends strongly on rhythm, groove, arrangement, or stylistic overlap.

The experimental design was implemented as planned. The project includes feature extraction, duplicate removal, preprocessing, baseline model comparison, primary model optimization, held-out test evaluation, cross-validation, error analysis, permutation importance, learning-curve analysis, and hyperparameter stability analysis. These elements form a coherent sequence: the data representation defines the model space, the baselines establish reference performance, the RBF SVM tests a stronger non-linear classifier, and the final analysis interprets both the improvement and the remaining weaknesses.

| Experiment | Purpose | Contribution to the research objective |
|---|---|---|
| Dummy classifier | Chance-level reference | Confirms that learned models are meaningful |
| Logistic regression | Linear baseline | Tests linear separability |
| Decision tree | Simple non-linear baseline | Tests low-complexity non-linear splits |
| Random forest | Strong tabular baseline | Establishes the strongest baseline |
| Gradient boosting | Additional ensemble baseline | Extends baseline comparison |
| RBF SVM | Primary model | Tests the final non-linear classifier |
| SelectKBest variant | Feature-selection check | Provides a useful negative result |
| Error analysis | Class-level diagnosis | Identifies weak genres and failure modes |
| Permutation importance | Interpretability | Connects model behavior to audio features |

No major experiment is disconnected from the central research objective. The SelectKBest variant did not improve the result, but it still contributes scientifically because it prevents an unsupported claim that feature selection improves generalization.

## B. Completeness of Experimental Evidence

The experimental evidence is sufficient for the engineering thesis scope. Baselines are properly implemented and compared under a shared protocol. The baseline stage includes a dummy classifier, logistic regression, decision tree, random forest, and gradient boosting. These models establish a performance ladder from trivial prediction to stronger non-linear tabular methods. The strongest baseline is the default random forest, with test accuracy of 0.6919 and macro F1 of 0.6892.

The primary model evaluation is also sufficient. The selected final model is an RBF SVM using all engineered features, with best parameters `C=3` and `gamma="scale"`. It achieved mean cross-validation accuracy of 0.7275, cross-validation standard deviation of 0.0130, test accuracy of 0.7121, and test macro F1 of 0.7121. The held-out test result is close to the cross-validation score, so there is no evidence of severe final-test collapse.

The comparison between the primary model and the strongest baseline is valid because both use the same cleaned dataset, target variable, random seed, split strategy, cross-validation design, preprocessing logic, and evaluation metrics. The SVM improves over the strongest baseline by approximately 0.0202 accuracy and 0.0228 macro F1. This is a real but moderate improvement. Therefore, the final conclusion should be that the SVM is the best tested model under the current protocol, not that it is a decisive breakthrough.

Error analysis and interpretability are adequately developed. Assignment 9 identifies class-level weaknesses and misclassification patterns. The model performs better on genres such as classical, jazz, pop, blues, and metal, while disco, rock, country, and reggae remain more difficult. Permutation importance is appropriate because the RBF SVM has no direct coefficients. The most important features are musically meaningful, especially MFCC and energy-related descriptors, but this analysis should be treated as broad feature-group evidence rather than detailed causal explanation.

The main weaknesses are:

- the improvement over the strongest baseline is moderate;
- the final SVM has a substantial train-validation gap of about 0.2585;
- the test set is relatively small for strong class-level claims;
- global audio summaries do not fully capture temporal structure;
- no deep learning or spectrogram-based model is evaluated, which limits general conclusions.

These weaknesses do not invalidate the work. They define the correct boundaries of the thesis claims and should be explicitly reflected in the final conclusion.

## C. Internal Consistency of Writing

The writing is mostly internally consistent. The same core terms are used across the practical component: baseline models, primary model, final RBF SVM, held-out test set, cross-validation, macro F1, engineered features, and global audio descriptors. The project also follows a clear sequence: Assignment 6 defines preprocessing, Assignment 7 establishes baselines, Assignment 8 optimizes the primary model, Assignment 9 interprets results, and Assignment 10 consolidates methodology and implementation.

The key numerical values are consistent across the main chapters. The final SVM is reported with 0.7121 test accuracy and 0.7121 macro F1. The strongest baseline, default random forest, is reported with 0.6919 test accuracy and 0.6892 macro F1. The dataset size after duplicate removal is consistently described as 987 observations, after removing 13 exact duplicate rows from the original 1000 observations.

The methodological descriptions are clear. The thesis explains why accuracy is used as the primary metric, why macro metrics are needed, why preprocessing is placed inside pipelines, and why an RBF SVM is a reasonable primary model for a small tabular audio-feature dataset. The interpretation follows logically from the results and avoids unsupported claims.

The main writing issue is possible redundancy between Assignment 9 and Assignment 10. In the final thesis, the methodology chapter should explain the protocol, the implementation chapter should explain the technical realization, and the results chapter should contain the full interpretation. Repeated metric tables should be merged or given clearly different functions.

All major claims are supported by evidence. The claim that the final model exceeds the hypothesis threshold is supported by the 0.7121 test accuracy. The claim that the improvement is moderate is supported by the approximately two-percentage-point gain over random forest. The claim that the model has overfitting risk is supported by the high training accuracy and train-validation gap. The claim that global features are limited is supported by class-level error patterns and feature-importance interpretation.

## D. Technical Integrity and Reproducibility Check

The repository reflects the reported experimental pipeline. The baseline stage is implemented in `src/baseline_models.py`, the primary model stage in `src/target_model.py`, and the final analytical stage in `src/results_analysis.py`. The scripts load `data/features_extended.csv` and save result artifacts to the `docs` directory.

The main reproducibility controls are present. The scripts use `RANDOM_STATE = 30`, `TEST_SIZE = 0.2`, and 5-fold stratified cross-validation. The train-test split is stratified by genre. Scaling and model fitting are performed inside scikit-learn pipelines, which reduces leakage risk. Duplicate removal is performed before splitting, which prevents exact duplicate rows from appearing in both training and test data.

The generated outputs correspond to the thesis evidence: baseline results, target-model results, comparative results, optimization results, learning curve, classification reports, confusion matrices, consolidated results, error summaries, misclassification pairs, permutation importance, and hyperparameter stability.

Before final submission, the repository still requires cleanup. Generated `__pycache__` directories should not be part of the final repository state. The modified `src/experiment_best_accuracy.py` file should be reviewed. If it is not part of the final experimental pipeline, it should not influence the thesis narrative. If it is retained, its role should be documented.

| Check | Status |
|---|---|
| Shared dataset and target variable | Satisfied |
| Duplicate removal | Satisfied |
| Stratified split and fixed seed | Satisfied |
| 5-fold stratified CV | Satisfied |
| Pipeline-based preprocessing | Satisfied |
| Output CSV files for reported results | Satisfied |
| Removal of temporary cache files | Required before submission |
| Review of modified auxiliary script | Required before submission |
| Final metric check against CSV outputs | Required before submission |

## E. Identification of Necessary Corrections

The required corrections before final submission are:

| Correction | Priority | Reason |
|---|---|---|
| Verify all final metrics against CSV outputs | Required | Protects scientific validity |
| Remove generated `__pycache__` artifacts | Required | Protects repository integrity |
| Keep final conclusions cautious | Required | Maintains logical coherence |
| Reduce redundancy between results and methodology chapters | Required | Improves unity of the thesis |

The optional improvements are:

| Improvement | Priority | Reason |
|---|---|---|
| Add a compact research question-method-evidence-conclusion table | Optional | Strengthens logical continuity |
| Clarify the negative SelectKBest result | Optional | Shows critical interpretation |
| Add a short limitations subsection before final conclusions | Optional | Prepares Assignment 12 |
| Standardize wording for "primary model" and "final model" | Optional | Improves terminology consistency |

The non-critical refinements are:

| Refinement | Priority |
|---|---|
| Check table formatting and captions | Low |
| Check figure and table references | Low |
| Standardize decimal formatting | Low |
| Remove repeated sentences | Low |
