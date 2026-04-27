# Assignment 7 - Baseline Models

## 1. Objective and role in the project

The objective of this assignment is to establish a controlled empirical baseline for the music genre classification project. The task is a multi-class classification problem in which each audio track is represented by 61 numerical features extracted with `librosa`, and the target variable is the music genre. The dataset contains ten classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

The baseline stage is important because later, more complex models must be evaluated against a clear reference point. A model will be considered useful only if it improves over the baseline under the same data split, preprocessing rules, and evaluation protocol. The goal is therefore not only to obtain the highest possible score, but to define a fair and reproducible comparison framework.

The implementation is provided in `src/baseline_models.py`. The structured outputs are saved as:

- `docs/assignment7_baseline_results.csv`
- `docs/assignment7_classification_reports.csv`
- `docs/assignment7_best_confusion_matrix.csv`

## 2. Dataset and preprocessing protocol

The experiment uses `data/features_extended.csv`, not the globally scaled exploratory file. This is consistent with Assignment 6, where global scaling was identified as unsuitable for final modeling because it can leak information from the test set into the training process.

The dataset before cleaning contains 1000 observations and 62 columns: 61 numerical predictors and the target column `genre`. The class distribution is balanced, with 100 recordings per genre. Before modeling, 13 exact duplicate rows were removed. The final experimental dataset therefore contains 987 observations and 61 predictors. No missing values were detected.

All models use the same controlled protocol:

| Element | Configuration |
|---|---|
| Input data | `data/features_extended.csv` |
| Rows after duplicate removal | 987 |
| Number of predictors | 61 |
| Target variable | `genre` |
| Test split | 20% |
| Split type | Stratified train-test split |
| Random seed | 30 |
| Cross-validation | 5-fold stratified CV on the training set |
| Primary metric | Accuracy |
| Supporting metrics | Macro precision, macro recall, macro F1 |
| Leakage control | Scaling fitted inside the pipeline only on training folds |

The same split was used for every model. Hyperparameter tuning was performed only inside the training set using cross-validation. The test set was used only once per fitted configuration for final evaluation. This prevents test-set contamination and keeps the comparison fair.

All classifiers were evaluated through a scikit-learn `Pipeline` containing `StandardScaler` and the model. Scaling is essential for logistic regression and does not introduce methodological unfairness for tree models because it is applied identically within the pipeline and fitted only on training data. The important control condition is that no preprocessing step is fitted on the full dataset before splitting.

## 3. Baseline model selection and justification

The baseline set includes one naive reference model, one simple linear model, and several classical machine learning models. This gives a realistic empirical ladder: performance must first exceed a trivial class-frequency baseline, then a simple interpretable model, and then classical non-linear methods.

| Model | Role in baseline | Rationale | Expected strengths | Expected weaknesses |
|---|---|---|---|---|
| Dummy classifier | Naive reference | Predicts the most frequent class and verifies that learned models outperform chance-level behavior. | Simple sanity check; useful lower bound. | Cannot use audio features; expected accuracy near 10% because the dataset is balanced. |
| Logistic regression | Simple model | Represents a minimal linear baseline for multi-class classification on standardized audio features. | Fast, transparent, strong when classes are linearly separable in MFCC/chroma/spectral space. | Limited ability to model non-linear relations and interactions between audio descriptors. |
| Decision tree | Classical ML model | Provides an interpretable non-linear baseline using feature thresholds. | Captures interactions and non-linear splits; easy to interpret. | High variance and overfitting risk on a relatively small dataset. |
| Random forest | Classical ensemble baseline | Standard robust tabular baseline that averages many decision trees. | Reduces variance, handles non-linear feature interactions, usually strong on engineered tabular features. | Less interpretable than a single tree; may not exploit temporal audio structure. |
| Gradient boosting | Classical ensemble baseline | Sequential tree ensemble often effective for structured/tabular feature sets. | Can model complex decision boundaries and correct previous errors. | Sensitive to hyperparameters and may overfit if tuned too aggressively. |

This selection is consistent with the project context described in Assignment 6. The features are engineered summary descriptors rather than raw waveforms, so classical machine learning models are appropriate baselines before moving to more complex audio models or heavily optimized SVM/neural approaches.

## 4. Hyperparameter configuration strategy

Two configuration types were distinguished:

- Default configurations: direct model defaults with only reproducibility parameters fixed where applicable.
- Tuned configurations: limited grid search over a realistic engineering search space.

The tuning strategy was intentionally conservative. Baselines should not be over-optimized to the point where they become the primary model. The purpose is to estimate whether basic model families are competitive and to provide a fair reference for later experiments.

| Model | Default configuration | Tuned search space |
|---|---|---|
| Logistic regression | `max_iter=5000`, `random_state=30` | `C=[0.1, 1.0, 10.0]`, `solver=["lbfgs"]` |
| Decision tree | `random_state=30` | `max_depth=[5, 10, None]`, `min_samples_leaf=[1, 3, 5]` |
| Random forest | `random_state=30`, `n_jobs=-1` | `n_estimators=[200, 500]`, `max_depth=[None, 15]`, `max_features=["sqrt", "log2"]` |
| Gradient boosting | `random_state=30` | `n_estimators=[100, 200]`, `learning_rate=[0.05, 0.1]`, `max_depth=[2, 3]` |

The tuning score was accuracy, matching the primary metric. Because the classes are balanced, accuracy is an appropriate primary measure. Macro metrics were also reported to avoid hiding weak class-level performance.

## 5. Performance results

The table below reports the main results. Cross-validation metrics are calculated on the training set. Test metrics are calculated once on the held-out test set.

| Model | Config. | CV accuracy mean | CV accuracy std | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Random forest | Default | 0.6692 | 0.0309 | 0.6919 | 0.6892 |
| Random forest | Tuned | 0.6806 | 0.0166 | 0.6818 | 0.6795 |
| Gradient boosting | Tuned | 0.6388 | 0.0129 | 0.6465 | 0.6470 |
| Gradient boosting | Default | 0.6337 | 0.0221 | 0.6313 | 0.6325 |
| Logistic regression | Default | 0.6831 | 0.0339 | 0.6111 | 0.6061 |
| Logistic regression | Tuned | 0.6831 | 0.0339 | 0.6111 | 0.6061 |
| Decision tree | Default | 0.4309 | 0.0214 | 0.4495 | 0.4464 |
| Decision tree | Tuned | 0.4385 | 0.0311 | 0.4444 | 0.4394 |
| Dummy classifier | Default | 0.1014 | 0.0003 | 0.1010 | 0.0183 |

The best test-set result was obtained by the default random forest, with accuracy 0.6919 and macro F1 0.6892. The tuned random forest had a slightly higher cross-validation mean but a slightly lower test accuracy. This difference is small, but it is methodologically important: the final ranking should be based on the held-out test result, while CV results should be used to understand stability and tuning behavior.

For the best model, the class-level report was:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| blues | 0.8571 | 0.9000 | 0.8780 |
| classical | 0.8696 | 1.0000 | 0.9302 |
| country | 0.5789 | 0.5500 | 0.5641 |
| disco | 0.6250 | 0.5000 | 0.5556 |
| hiphop | 0.7333 | 0.5500 | 0.6286 |
| jazz | 0.8947 | 0.8500 | 0.8718 |
| metal | 0.7895 | 0.8333 | 0.8108 |
| pop | 0.6250 | 0.7500 | 0.6818 |
| reggae | 0.5455 | 0.6000 | 0.5714 |
| rock | 0.4000 | 0.4000 | 0.4000 |

The confusion matrix confirms that the model handles classical, blues, jazz, and metal relatively well. The weakest class is rock, which is frequently confused with country, disco, metal, and other rhythmically or timbrally related genres. This is a plausible result for a feature set based on global statistics, because rock overlaps with several neighboring genres in spectral energy, tempo, and timbre.

## 6. Critical interpretation

The baseline results show that the extracted audio features contain useful genre information. Every learned model strongly outperformed the dummy classifier. The dummy model achieved approximately 10% accuracy, which is expected for a balanced ten-class problem. The best learned baseline reached approximately 69% accuracy, so the improvement is substantial and cannot be explained by class imbalance.

The strongest model was the random forest. This result is reasonable for the current representation. The dataset consists of engineered tabular descriptors such as MFCC means and standard deviations, chroma statistics, RMS energy, zero crossing rate, and spectral descriptors. Random forest can combine these features through non-linear decision rules and is less vulnerable than a single decision tree to high variance. Its good performance suggests that genre separability is partly non-linear and depends on interactions between timbral, harmonic, and spectral attributes.

The single decision tree performed much worse, around 44-45% accuracy. This is also expected. A single tree is flexible but unstable; it can fit local thresholds that do not generalize well. The gap between decision tree and random forest confirms that ensembling is beneficial for this dataset.

Gradient boosting performed competitively but did not exceed random forest. The tuned version improved over the default version, from 0.6313 to 0.6465 test accuracy, but the improvement was not enough to make it the strongest baseline. This suggests that sequential boosting can exploit the feature space, but the current limited tuning did not produce a decisive advantage. More aggressive tuning might improve the result, but that would shift the model away from a baseline role and should be reserved for later primary-model optimization.

Logistic regression produced an important methodological signal. Its cross-validation accuracy was high, around 0.6831, but its test accuracy was lower, 0.6111. This discrepancy may indicate that the linear boundary is not stable across the final test split, or that some genre distinctions require non-linear interactions. Since logistic regression and tuned logistic regression selected the same effective configuration, the limited hyperparameter search did not change the model behavior. The result supports the conclusion that a purely linear baseline is not sufficient for the final system.

The class-level results also matter. The model performs very well on classical, blues, jazz, and metal, which likely have distinctive timbral or spectral profiles. It struggles more with rock, country, disco, and reggae. This aligns with the expectation from Assignment 6 that global summary features compress temporal structure. Genres that share instrumentation, rhythm patterns, or production characteristics may become difficult to separate when each track is represented only by means and standard deviations.

## 7. Relation to the research question

The research question concerns whether music genre can be recognized automatically from extracted audio features using machine learning methods. The baseline stage answers this question partially and empirically: yes, the extracted features are informative enough to support classification substantially above chance level. However, the baseline also shows that the current representation and classical models do not solve the task completely.

The best baseline accuracy of 0.6919 establishes the minimum empirical threshold for later models. Any advanced model, such as a more carefully optimized SVM, a model with feature selection, or a neural model operating on richer audio representations, should be compared against this number under the same split and evaluation protocol. Improvements should be judged not only by accuracy, but also by macro F1 and by whether weak classes such as rock, country, disco, and reggae improve.

## 8. Conclusion from the baseline stage

The baseline stage provides three main conclusions.

First, the experimental pipeline is valid: preprocessing is fitted inside the model pipeline, the test set remains untouched during tuning, random seeds are fixed, and all models use the same cleaned dataset and stratified split.

Second, classical models provide a meaningful reference point. The best baseline is the default random forest with 0.6919 test accuracy and 0.6892 macro F1. This is strong enough to show that the feature extraction strategy is useful, but not strong enough to make further modeling unnecessary.

Third, moving toward more complex models is justified. The remaining errors are not random; they concentrate in musically overlapping genres. This indicates that future work should focus on richer modeling of feature interactions, improved temporal representation, or carefully controlled feature selection and model optimization. From this point forward, any claim of improvement should be benchmarked against the random forest baseline reported in this assignment.
