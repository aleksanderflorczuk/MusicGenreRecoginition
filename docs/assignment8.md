# Assignment 8 - Target Model and Optimization

## 1. Objective and role in the project

The objective of this assignment is to implement and evaluate the primary model for the music genre classification project. Assignment 7 established a controlled baseline under a fixed experimental protocol. The strongest baseline was the default random forest, which achieved 0.6919 test accuracy and 0.6892 macro F1 on the held-out test set. Assignment 8 therefore evaluates whether a more specialized model can improve performance while preserving the same methodological constraints.

The task remains a ten-class music genre classification problem. Each recording is represented by 61 numerical audio descriptors extracted with `librosa`, including tempo, zero crossing rate, RMS energy, spectral descriptors, MFCC statistics, and chroma statistics. The target variable is `genre`, with the following classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

The implementation is provided in `src/target_model.py`. The experiment outputs are saved as:

- `docs/assignment8_target_results.csv`
- `docs/assignment8_comparative_results.csv`
- `docs/assignment8_optimization_results.csv`
- `docs/assignment8_learning_curve.csv`
- `docs/assignment8_classification_report.csv`
- `docs/assignment8_best_confusion_matrix.csv`

## 2. Justification of model selection

The selected primary model is a support vector machine with a radial basis function kernel:

```text
StandardScaler -> SVC(kernel="rbf")
```

An additional experimental variant was evaluated:

```text
StandardScaler -> SelectKBest(f_classif) -> SVC(kernel="rbf")
```

The RBF SVM was selected because it directly addresses the main limitation observed in Assignment 7. Logistic regression showed that a purely linear decision boundary was not sufficient: despite high cross-validation accuracy, its held-out test accuracy was only 0.6111. A single decision tree also performed weakly because it was unstable. Random forest improved performance by using non-linear feature interactions, but its remaining errors were concentrated in overlapping genres such as rock, country, disco, and reggae.

The RBF SVM is therefore a conceptually grounded target model. It is not selected because it is generally popular, but because it matches the structure of the current feature representation. The dataset is relatively small, balanced, fully numerical, and standardized. In this setting, a margin-based classifier with a non-linear kernel can learn smooth class boundaries in the engineered feature space without requiring a deep architecture or raw audio input. This is appropriate for an engineering thesis where the goal is controlled improvement rather than unnecessary architectural complexity.

The model also connects to the research question from Assignment 7: whether music genre can be recognized automatically from extracted audio features using machine learning methods. The baseline already showed that the extracted features are informative. The target model tests a more specific hypothesis: genre separation in this feature space can be improved by a non-linear margin-based classifier that is more flexible than logistic regression but more controlled than aggressively tuned ensembles or neural models.

The optional feature-selection variant was included because Assignment 6 identified correlation among some spectral descriptors and warned that automatic feature selection must be fitted only inside the training folds. The goal was to test whether reducing noisy or redundant predictors improves generalization. The result did not support feature removal, because the best selected value was `k="all"`. For this reason, the final primary model is the simpler all-feature RBF SVM.

## 3. Model architecture and configuration

The primary model is a scikit-learn pipeline with two components:

| Component | Configuration | Role |
|---|---|---|
| `StandardScaler` | default parameters | Fits feature means and standard deviations on the training fold only; transforms train, validation, and test data consistently. |
| `SVC` | `kernel="rbf"` | Learns non-linear class boundaries using a radial basis function kernel. |

The input interface is a tabular matrix with 61 numerical predictors. The output interface is a predicted genre label from one of ten classes. The model receives no raw waveform data and no categorical predictors.

The core hyperparameters are:

| Hyperparameter | Meaning | Search values |
|---|---|---|
| `C` | Controls regularization strength. Larger values reduce margin violations but increase overfitting risk. | `[1, 3, 10, 30, 100]` |
| `gamma` | Controls the locality of the RBF kernel. Larger values create more local and potentially more complex boundaries. | `["scale", 0.03, 0.01, 0.003, 0.001]` |
| `selector__k` | Number of retained features in the optional feature-selection variant. | `[20, 30, 40, 50, "all"]` |

The architectural rationale is conservative. The project currently uses global summary features, not spectrograms or time sequences. A neural network operating on this 987-row table would add capacity without a strong data-volume justification. The RBF SVM adds meaningful non-linearity while keeping the model small enough for exhaustive grid search and transparent validation.

## 4. Optimization strategy

Hyperparameter optimization was performed with `GridSearchCV` using 5-fold stratified cross-validation on the training set only. The test set was not used during tuning. This is consistent with Assignment 3 and Assignment 7, where the split strategy was fixed as:

| Element | Configuration |
|---|---|
| Input data | `data/features_extended.csv` |
| Duplicate handling | 13 exact duplicate rows removed before splitting |
| Final dataset size | 987 rows |
| Number of predictors | 61 |
| Test split | 20% |
| Split type | Stratified train-test split |
| Random seed | 30 |
| Cross-validation | 5-fold stratified CV on the training set |
| Primary metric | Accuracy |
| Supporting metrics | Macro precision, macro recall, macro F1 |

The search space was intentionally bounded. The goal was not to maximize the score through excessive tuning, but to evaluate a plausible range of margin and kernel-width settings. The grid included both moderate and high values of `C`, plus several gamma values. This allowed the experiment to detect whether stronger regularization or smoother boundaries generalized better.

Overfitting detection was implemented through:

- `return_train_score=True` in `GridSearchCV`;
- comparison of training and validation accuracy;
- cross-validation standard deviation across folds;
- a learning curve for the selected model;
- final evaluation on the untouched test split.

The selected all-feature RBF SVM used:

```text
SVC(kernel="rbf", C=3, gamma="scale")
```

The optional feature-selection pipeline selected:

```text
SVC(kernel="rbf", C=3, gamma="scale"), selector__k="all"
```

Because both variants produced the same validation and test performance, the all-feature model is preferred. It is simpler, faster, and avoids an unnecessary feature-selection step.

## 5. Optimization results and sensitivity

The best validation configurations were:

| Rank | Experiment | Parameters | Mean train accuracy | Mean validation accuracy | Validation std | Generalization gap |
|---:|---|---|---:|---:|---:|---:|
| 1 | SVM RBF with feature selection | `C=3`, `gamma="scale"`, `k="all"` | 0.9861 | 0.7275 | 0.0130 | 0.2585 |
| 1 | SVM RBF all features | `C=3`, `gamma="scale"` | 0.9861 | 0.7275 | 0.0130 | 0.2585 |
| 3 | SVM RBF all features | `C=3`, `gamma=0.01` | 0.9541 | 0.7263 | 0.0153 | 0.2278 |
| 3 | SVM RBF with feature selection | `C=3`, `gamma=0.01`, `k="all"` | 0.9541 | 0.7263 | 0.0153 | 0.2278 |
| 5 | SVM RBF with feature selection | `C=10`, `gamma=0.01`, `k=50` | 0.9857 | 0.7262 | 0.0238 | 0.2595 |

The sensitivity analysis shows that the model is fairly stable around the best region. The difference between the best configuration and `C=3, gamma=0.01` is only about 0.0013 validation accuracy. This means the improvement is not caused by a single fragile hyperparameter value. However, larger `C` values increased training accuracy toward almost perfect memorization without improving validation accuracy. For example, `C=10, gamma="scale"` reached 0.9990 training accuracy but only 0.7199 validation accuracy. This confirms that stronger capacity leads to diminishing returns.

The feature-selection variant did not provide an advantage. Its best result retained all features, and the full grid took longer to run. Therefore, feature selection is not included in the final target model. This is an important engineering conclusion: the correct decision is not to keep a more complex pipeline when it does not improve generalization.

## 6. Generalization and overfitting analysis

The selected model achieved:

| Metric | Value |
|---|---:|
| Mean CV accuracy | 0.7275 |
| CV accuracy standard deviation | 0.0130 |
| Mean train accuracy in CV | 0.9861 |
| Train-validation gap | 0.2585 |
| Test accuracy | 0.7121 |
| Test macro F1 | 0.7121 |

The train-validation gap is substantial. This means the RBF SVM fits the training folds very strongly, which is expected for a non-linear kernel model. However, the validation score remains stable across folds, with a relatively small standard deviation of 0.0130. The held-out test score of 0.7121 is also reasonably close to the mean cross-validation score of 0.7275. The difference between CV and test accuracy is approximately 0.0154, which does not suggest severe final-test collapse.

The learning curve gives additional context:

| Training size | Train accuracy | Validation accuracy | Generalization gap |
|---:|---:|---:|---:|
| 126 | 1.0000 | 0.6045 | 0.3955 |
| 252 | 0.9984 | 0.6654 | 0.3330 |
| 378 | 0.9942 | 0.6894 | 0.3047 |
| 504 | 0.9861 | 0.6984 | 0.2877 |
| 631 | 0.9857 | 0.7275 | 0.2582 |

Validation accuracy improves consistently as more training data are used, while the generalization gap narrows. This suggests that the model benefits from additional data and that the remaining overfitting is partly related to limited sample size rather than only poor hyperparameter choice. The curve does not show a clear validation plateau at small training sizes; the largest training size produces the best validation score. This supports the use of the full training set for final fitting.

At the same time, the high training accuracy means the result should be interpreted with restraint. The model improves over the baseline, but it does not eliminate the fundamental limitation of the current representation: each song is compressed into global summary statistics. Genre distinctions that depend on temporal progression, rhythm patterns, or local arrangement changes remain difficult.

## 7. Comparative evaluation against baselines

The primary comparison is against the strongest Assignment 7 baseline, the default random forest. Both models use the same input file, duplicate-removal rule, stratified split, random seed, and test set.

| Model | Role | CV accuracy | Test accuracy | Test macro F1 | Fit/tuning time | Test prediction time |
|---|---|---:|---:|---:|---:|---:|
| Random forest default | Best Assignment 7 baseline | 0.6692 | 0.6919 | 0.6892 | 1.06 s | 0.027 s |
| SVM RBF all features | Assignment 8 primary model | 0.7275 | 0.7121 | 0.7121 | 8.10 s | 0.017 s |

The absolute improvement in test accuracy is:

```text
0.7121 - 0.6919 = 0.0202
```

The relative test accuracy improvement is:

```text
0.0202 / 0.6919 = 2.92%
```

The absolute improvement in macro F1 is:

```text
0.7121 - 0.6892 = 0.0228
```

The relative macro F1 improvement is:

```text
0.0228 / 0.6892 = 3.31%
```

The gain is meaningful but moderate. The SVM improves both accuracy and macro F1, which is important because macro F1 is sensitive to class-level weakness. The computational cost is higher during tuning: the target model grid took about 8.10 seconds, while the default random forest reference took about 1.06 seconds in the same script. Prediction time on the small test split was similar and not practically limiting.

The class-level report for the selected SVM is:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| blues | 0.7619 | 0.8000 | 0.7805 |
| classical | 0.9048 | 0.9500 | 0.9268 |
| country | 0.5909 | 0.6500 | 0.6190 |
| disco | 0.4500 | 0.4500 | 0.4500 |
| hiphop | 0.8667 | 0.6500 | 0.7429 |
| jazz | 0.8500 | 0.8500 | 0.8500 |
| metal | 0.7647 | 0.7222 | 0.7429 |
| pop | 0.8095 | 0.8500 | 0.8293 |
| reggae | 0.6364 | 0.7000 | 0.6667 |
| rock | 0.5263 | 0.5000 | 0.5128 |

Compared with the random forest baseline, the SVM improved several previously weak or moderate classes. Rock F1 increased from 0.4000 to 0.5128, country F1 increased from 0.5641 to 0.6190, hiphop F1 increased from 0.6286 to 0.7429, pop F1 increased from 0.6818 to 0.8293, and reggae F1 increased from 0.5714 to 0.6667. These improvements support the choice of a non-linear margin-based model.

The main exception is disco, where F1 decreased from 0.5556 to 0.4500. The confusion matrix shows that disco remains distributed across several neighboring genres, especially country, metal, pop, reggae, and rock. This suggests that the global feature representation still does not fully capture rhythm and arrangement characteristics needed to separate disco reliably.

## 8. Interpretation in relation to the research question

The target model supports the research hypothesis in a measured way. Music genre can be recognized from extracted audio features substantially above chance level, and a non-linear margin-based classifier improves over the strongest baseline under the same evaluation protocol.

The improvement is not large enough to justify exaggerated claims. The primary model does not solve the classification task completely, and several genre pairs remain difficult. However, the gain is consistent across the primary metrics: accuracy, macro precision, macro recall, and macro F1 all exceed the random forest baseline. The improvement is also methodologically credible because the test set remained untouched during tuning and all transformations were fitted inside the pipeline.

The result implies that the research gap identified in the earlier assignments is real but bounded. Classical baselines can capture much of the information in global audio descriptors, but a more specialized non-linear classifier can extract additional structure from the same features. The remaining errors indicate that further improvement would likely require richer audio representation, such as temporal features, beat-synchronous descriptors, or spectrogram-based modeling, rather than only more aggressive tuning of the same feature table.

## 9. Conclusion

Assignment 8 completes the core experimental phase by introducing and validating the primary model. The selected model is an RBF SVM with standardized input features and tuned `C` and `gamma`. It achieved 0.7121 test accuracy and 0.7121 macro F1, improving over the best Assignment 7 baseline by 0.0202 absolute accuracy and 0.0228 absolute macro F1.

The optimization was controlled and transparent. Hyperparameters were selected with stratified cross-validation on the training set only, preprocessing was contained inside the pipeline, and the held-out test set was used only for final evaluation. The analysis also shows clear signs of model capacity and overfitting risk, so the result should be interpreted as a moderate but defensible engineering improvement rather than a definitive solution.

The practical component is now functionally complete. Further work should focus on scientific interpretation, clearer thesis writing, and carefully justified incremental improvements rather than unstable experimentation.
