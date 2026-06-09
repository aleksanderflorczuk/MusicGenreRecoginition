# Assignment 2: Systematic Literature Review

**Topic:** Music Genre Recognition Using Machine Learning

---

## Introduction and Research Context

This literature review positions the thesis problem within research on automatic music genre recognition (MGR). The thesis investigates whether music genre can be recognized automatically from engineered audio descriptors using classical machine learning, and whether a non-linear support vector machine provides a measurable improvement over simpler baseline models.

The experimental problem is formulated as supervised ten-class classification. Each recording belongs to one of the following categories: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, or rock. A recording is represented by a fixed-length vector containing tempo, RMS energy, zero-crossing rate, spectral centroid, spectral bandwidth, spectral rolloff, mel-frequency cepstral coefficient (MFCC) statistics, and chroma statistics.

This formulation belongs to the established content-based MGR paradigm introduced by Tzanetakis and Cook [1]. However, later research shows that high classification accuracy alone does not establish that a model has learned general musical properties of genre. Results may be influenced by duplicated recordings, repeated artists, recording conditions, encoding artifacts, ambiguous labels, or an evaluation split that allows related material to occur in both training and test sets [5], [6].

The purpose of this review is therefore not to identify the highest published GTZAN score. Its purpose is to determine:

- which feature representations and classifier families dominate MGR research;
- how datasets and evaluation procedures affect reported results;
- whether classical engineered descriptors remain methodologically justified;
- which weaknesses support the controlled comparison performed in this thesis.

---

## A. Search Strategy and Source Selection

### A.1. Databases and Repositories

The search covered sources indexed or published through:

- **IEEE Xplore** - technical publications in signal processing and machine learning.
- **ACM Digital Library** - computer science conference and journal publications.
- **SpringerLink** - peer-reviewed journals and academic books.
- **ISMIR Proceedings Archive** - research focused on Music Information Retrieval.
- **Transactions of the International Society for Music Information Retrieval** - peer-reviewed MIR journal articles.
- **arXiv** - accessible versions of papers and additional preprints.
- **MIREX and MediaEval archives** - benchmark protocols and comparative results.
- **Official FMA resources** - dataset documentation and benchmark information.

Searches were conducted for publications available up to June 2026. The main scientific literature was concentrated in the period 2002-2019 because this interval covers the introduction of the GTZAN benchmark, the development of classical feature-based systems, the transition toward learned representations, and major methodological criticism of conventional MGR evaluation. Newer publications were screened where relevant, but topical recency alone was not treated as a selection criterion.

### A.2. Keywords and Search Queries

The principal search strings were:

```text
"music genre classification" AND audio AND MFCC
"music genre recognition" AND support vector machine
GTZAN AND genre classification AND evaluation
GTZAN AND duplicates OR artist filter OR faults
"music classification" AND AdaBoost AND aggregate features
"music genre" AND transfer learning AND convolutional network
"music genre classification" AND data augmentation
MIREX AND "audio genre classification"
MediaEval AND AcousticBrainz AND genre
FMA AND genre recognition AND benchmark
```

Backward reference searching was applied to the selected survey and evaluation papers. Forward searching was used for the foundational GTZAN paper and for publications that questioned the validity of standard evaluation practice.

### A.3. Inclusion Criteria

A source was included when it satisfied at least one of the following conditions:

- introduced or evaluated an audio-based MGR method;
- compared feature representations or classifier families relevant to the thesis;
- used GTZAN, ISMIR2004, FMA, or another recognized MGR benchmark;
- analyzed evaluation validity, dataset faults, confounding, or reproducibility;
- defined a benchmark protocol relevant to fair model comparison.

Scientific papers had to be peer-reviewed journal or conference publications. An arXiv version was accepted as an access route when the work was also published in a recognized venue. One additional non-peer-reviewed GTZAN analysis was retained only as supplementary methodological evidence and was not counted toward the required minimum of peer-reviewed papers. Benchmark reports were accepted when produced by established initiatives such as MIREX, MediaEval, or the FMA challenge.

### A.4. Exclusion Criteria

The review excluded:

- tutorials, personal blogs, and unsourced implementation articles;
- papers that classified only symbolic MIDI data or lyrics without an audio experiment;
- studies without enough methodological information to identify the representation, classifier, dataset, and evaluation procedure;
- duplicate versions of the same study;
- results based only on private datasets when they offered no transferable methodological insight;
- claims of very high accuracy that could not be interpreted because the split or leakage controls were unspecified.

### A.5. Time Range and Final Source Scope

The search considered sources published from 2002 to June 2026. The final core set contains eight peer-reviewed scientific papers and three distinct benchmark reports: MIREX, the FMA Challenge, and MediaEval. The peer-reviewed FMA paper additionally provides dataset documentation. One non-peer-reviewed GTZAN analysis was retained as supplementary methodological evidence because it directly documents dataset faults relevant to the thesis protocol. The selection therefore exceeds the minimum requirement of five scientific papers and meets the requirement of three benchmark reports. It emphasizes methodological relevance to a small, feature-based classification experiment rather than superficial similarity in title.

| Source category | Required minimum | Included scope |
|---|---:|---:|
| Peer-reviewed scientific papers | 5 | 8 |
| Benchmark reports | 3-5 | 3 |
| Supplementary methodological preprint | Optional | 1 GTZAN analysis |

---

## B. Thematic Organization and Synthesis of Literature

### B.1. Engineered Audio Descriptors and Classical Classifiers

The foundational approach represents each recording through manually designed descriptors of timbre, rhythm, and pitch. Tzanetakis and Cook [1] combined timbral texture, rhythmic content, and pitch content features and reported 61% accuracy on the ten-class dataset later known as GTZAN. The importance of this work is not its present-day score, but its formulation of MGR as supervised classification from audio content.

Subsequent work improved this pipeline mainly in three ways: richer features, better temporal aggregation, and stronger classifiers. Bergstra et al. [2] aggregated frame-level features over longer segments and used AdaBoost for joint classification and feature selection. Their reported GTZAN accuracy of approximately 82.5% and strong MIREX performance indicated that summarizing short-time descriptors at song or segment level can be effective.

Panagakis, Benetos, and Kotropoulos [3] moved beyond simple summary statistics by extracting auditory spectro-temporal modulation representations and reducing their dimensionality with multilinear methods before SVM classification. Their comparison showed that non-negative tensor factorization and higher-order singular value decomposition were more effective than multilinear PCA. However, they also acknowledged that experimental differences prevented direct comparison across datasets and studies.

These studies establish a pattern relevant to the thesis: representation quality and aggregation strategy often matter as much as classifier choice. MFCC, spectral, rhythm, and pitch-related descriptors can support useful classification, but a fixed global vector loses the order and evolution of musical events. Means and standard deviations describe average timbre and variability, but they cannot directly model a rhythmic sequence, a structural transition, or the temporal placement of instrumentation.

SVM appears repeatedly because it is suitable for medium-dimensional representations and relatively small datasets. An RBF kernel can model non-linear class boundaries while remaining less data-demanding than a deep neural network trained from scratch. This provides a methodological basis for selecting RBF SVM as the thesis's primary model. At the same time, literature does not justify evaluating SVM in isolation. Linear, tree-based, ensemble, and naive baselines are necessary to determine whether any gain results from meaningful non-linearity rather than from an inadequately chosen reference.

### B.2. Learned Representations and Transfer Learning

Deep learning shifted attention from hand-crafted descriptors toward representations learned from spectrograms or waveforms. Choi et al. [4] trained a convolutional network on a large music-tagging source task and transferred intermediate activations to six target tasks. On GTZAN, the transferred convolutional representation achieved 89.8% accuracy, compared with 66.0% for their MFCC baseline. The result demonstrates that a representation learned from a larger external corpus can encode information not captured by conventional MFCC aggregation.

This result does not make classical models irrelevant for the current thesis. Transfer learning introduces additional training data, model complexity, computational requirements, and dependencies on the source task. A direct score comparison between a pre-trained convolutional representation and a model trained only on approximately one thousand local recordings would therefore be misleading. The relevant conclusion is narrower: temporal and hierarchical representations can improve genre discrimination, especially when sufficient external data are available.

FMA was created partly in response to the limited size and accessibility of earlier MIR datasets [7]. It contains more than 100,000 Creative Commons tracks, metadata, pre-computed features, predefined splits, and a hierarchical genre taxonomy. The FMA challenge later showed the value of open data and common evaluation splits for comparing submissions [11]. These resources represent a more realistic direction for large-scale or deep-learning research than repeatedly optimizing models on GTZAN.

For a small engineering project, however, a controlled classical pipeline remains defensible. It permits transparent preprocessing, feasible hyperparameter search, explicit baseline comparison, and feature-level interpretation. Its purpose should be framed as evaluating the capability and limitations of engineered descriptors, not competing directly with large-scale pre-trained systems.

### B.3. Dataset Scale, Augmentation, and Overfitting

Dataset size is a recurring methodological constraint. GTZAN contains only 1,000 excerpts, with 100 examples per class. This is convenient for reproducible teaching and small experiments, but it is insufficient for strong claims about the diversity of complete musical genres. The number of recordings is especially restrictive when model selection, feature selection, and hyperparameter tuning are performed on the same limited development data.

Mignot and Peeters [8] studied sound transformations and segmentation as augmentation strategies. Their experiments show that augmentation can improve robustness, particularly under cross-dataset mismatch, but that benefits depend on how and where transformations are applied. For example, transformations did not automatically improve in-dataset results, while selected augmentation improved one cross-dataset direction from 46.7% to 56.7%. Augmentation is therefore not a universal substitute for independent data.

This distinction matters for the thesis. Splitting every recording into many correlated segments can increase the number of rows without increasing the number of independent songs. If segments from one track occur in both training and test sets, the evaluation becomes optimistic. Augmentation and segmentation must therefore be grouped by source recording. Since the present thesis uses one global vector per recording, it avoids segment-level leakage but retains the limitation of a small effective sample size.

The literature also supports examining the train-validation gap and learning curves rather than reporting test accuracy alone. High training performance combined with materially lower cross-validation performance indicates that additional model capacity is fitting dataset-specific detail. In a small feature table, this risk applies both to flexible ensembles and to a tuned RBF SVM.

### B.4. Evaluation Methodology, Dataset Faults, and Confounding

Evaluation methodology is the most important critical theme in the selected literature. Sturm [5] argues that accuracy, precision, recall, and confusion matrices do not by themselves prove that an MGR system recognizes genre. A classifier may exploit non-musical or irrelevant regularities while still producing a high score. This criticism changes the interpretation of benchmark results: accuracy measures label prediction under a defined protocol, not general human-like understanding of genre.

Sturm's detailed analysis of GTZAN identified repetitions, mislabelings, distortions, and artist repetition [9]. These faults do not affect every system equally, so published scores obtained under different filtering and partitioning rules are not necessarily comparable. Later intervention-based analysis demonstrated that artist replication and other confounding properties can materially influence conclusions drawn from GTZAN [6].

Benchmark initiatives responded to similar concerns. MIREX standardized train-test tasks and common datasets, while later MIREX protocols explicitly required artist filtering for genre tasks so that training and test partitions contained different artists [10]. MediaEval's AcousticBrainz Genre Task addressed another limitation: the same recordings can receive different genre labels under AllMusic, Discogs, Last.fm, and other taxonomies [12]. The task therefore treated genre annotation as source-dependent rather than as a single unquestionable ground truth.

The practical implication is that published accuracy values must not be ranked without considering:

- whether folds are track-, album-, or artist-independent;
- whether duplicate or distorted recordings were removed;
- whether feature selection and scaling were fitted inside cross-validation;
- whether the test set influenced hyperparameter selection;
- whether labels were single-class, multi-label, or hierarchical;
- whether the same metric and class distribution were used;
- whether code, splits, and feature definitions were available.

The present thesis cannot eliminate every GTZAN limitation because reliable artist metadata are not available in the local feature table. It can, however, improve internal validity by removing exact duplicate rows before splitting, using one fixed stratified holdout, fitting transformations only within training folds, retaining an untouched test set, reporting macro metrics and class-level errors, and avoiding claims of cross-dataset generalization.

---

## C. Comparative Analytical Table

| Authors / year | Problem formulation | Dataset(s) | Methods applied | Evaluation metrics | Key results | Identified limitations |
|---|---|---|---|---|---|---|
| Tzanetakis and Cook, 2002 [1] | Single-label audio genre classification | Original ten-class collection later known as GTZAN | Timbral, rhythmic, and pitch features; statistical classifiers | Classification accuracy under the authors' protocol | 61% on the ten-class task | Small dataset; early protocol; genre treated as fixed single label |
| Bergstra et al., 2006 [2] | Genre and artist classification | GTZAN and MIREX collections | Aggregated frame features; AdaBoost with decision stumps | Accuracy; benchmark competition results | About 82.5% reported on GTZAN; first in MIREX 2005 genre task | Incomplete comparability of GTZAN split details; possible dataset confounds |
| Panagakis et al., 2008 [3] | Single-label genre classification | GTZAN; ISMIR2004 Genre | Spectro-temporal modulation tensors; NTF, HOSVD, MPCA; SVM | 10-fold stratified cross-validation accuracy | Best reported result: 78.20% on GTZAN and 80.95% on ISMIR2004 Genre | High-dimensional pipeline; small-sample problem; single-label assumption |
| Sturm, 2013 [5] | Evaluation validity of MGR systems | Literature survey and GTZAN system analyses | Behavioral analysis, confusion analysis, listening-based critique | Accuracy and system-behavior analysis | Shows that high classification accuracy need not demonstrate genre recognition | Primarily methodological critique rather than a new classifier |
| Choi et al., 2017 [4] | Transfer of learned audio representations | Six tasks including GTZAN | Pre-trained convolutional network features; target-task classifiers | Task-specific metrics; GTZAN accuracy | 89.8% on GTZAN versus 66.0% for the paper's MFCC baseline | Uses external source-task data; not directly comparable with local-only models |
| Defferrard et al., 2017 [7] | Open benchmark for music analysis and genre recognition | FMA, including small/medium/large subsets | Pre-computed features and baseline classifiers; predefined splits | Accuracy and additional task metrics depending on subset | Establishes a scalable open benchmark with 106,574 tracks | Genre hierarchy and imbalance make results less directly comparable with GTZAN |
| Rodriguez-Algarra et al., 2019 [6] | Measurement of confounding effects | GTZAN | Controlled interventions on data and partitions | Factorial analysis of classification outcomes | Confirms that artist replication and other factors can confound results | Analysis remains dataset-specific and requires metadata unavailable in many copies |
| Mignot and Peeters, 2019 [8] | Effect of audio augmentation on genre classification | ISMIR2004; 1517-Artists | Sound transformations, segmentation, standard descriptors, GMM | In-dataset and cross-dataset accuracy | Selected augmentation improved one cross-dataset result from 46.7% to 56.7% | Benefits depend on transformation and domain; synthetic samples are correlated |
| MIREX, 2005 [10] | Community benchmark for audio genre classification | Hidden or controlled MIREX genre collections | Multiple independently submitted systems | Shared data, metrics, and comparative significance tests | Demonstrated large differences among systems under a common benchmark | Datasets and systems changed across years; results are not universal rankings |
| MIREX, 2017 [10] | Artist-filtered train-test audio classification | US Pop and Latin genre tasks | Participant-defined systems under common execution protocol | Cross-validation with artist-independent folds | Formalizes artist filtering as an evaluation requirement | Taxonomies differ from the thesis and data are not identical to GTZAN |
| FMA Challenge, 2018 [11] | Open genre-recognition competition | FMA | Public submissions using audio and/or released features | Fixed challenge split and leaderboard metrics | Supports reproducible comparison on open, larger-scale data | Competition performance may depend on external resources and engineering budget |
| MediaEval AcousticBrainz, 2017 [12] | Genre recognition from multiple annotation sources | Large AcousticBrainz-derived sets | Baselines and participant systems on computed audio features | Source-specific hierarchical genre evaluation | Shows that genre ground truth varies materially by annotation source | Complex taxonomies and partially intersecting datasets complicate interpretation |

The table demonstrates why numerical results cannot be compared as if they came from one experiment. Dataset content, split unit, external pre-training, taxonomy, and leakage controls differ substantially. The reliable methodological trends are more important than the absolute ranking of reported scores.

---

## D. Critical Analysis and Research Gap Identification

### D.1. What Remains Insufficiently Addressed?

Three limitations recur across the literature.

First, many studies optimize representation and classifier jointly, making it difficult to determine whether improvement results from the model, the feature set, external training data, or the evaluation protocol. Deep transfer systems can outperform MFCC baselines, but they answer a different engineering question from a local-only classical comparison.

Second, GTZAN-based work often reports results under incompatible or insufficiently documented splits. Duplicate removal, artist filtering, feature scaling, model selection, and test-set isolation are not consistently described. A higher published score may therefore reflect a more favorable protocol rather than a more general method.

Third, aggregate accuracy dominates reporting even though genre classes overlap and class-specific behavior is uneven. Rock, country, disco, pop, and reggae frequently share timbral, rhythmic, or production characteristics. A model can achieve acceptable overall accuracy while remaining systematically weak for these classes.

### D.2. Where Do Existing Methods Fail or Underperform?

Global engineered descriptors underperform when genre identity depends on temporal development, groove, arrangement, or local instrumentation changes. Means and standard deviations remove the order of events. This can make two recordings with different musical structures appear similar if their global spectral distributions are alike.

Complex learned representations reduce this limitation but require more data and introduce weaker transparency. On a dataset of approximately one thousand tracks, training a deep model from scratch creates a substantial overfitting risk. Transfer learning reduces that risk but changes the resource assumptions and may transfer biases from the source corpus.

Existing evaluations also fail to establish broad generalization when they use one collection and random track-level splits. Without artist-independent or cross-dataset testing, the result applies only to the sampled distribution. This thesis must therefore state that its findings concern the selected dataset and representation.

### D.3. What Methodological Weakness Justifies the Proposed Approach?

The identified gap is not the absence of music genre classifiers. The problem has many proposed solutions. The defensible engineering gap is the need for a transparent, reproducible assessment of how far a compact set of global audio descriptors can support ten-class genre classification when:

- exact duplicate rows are removed before splitting;
- all models receive the same cleaned observations and split;
- preprocessing and feature selection are fitted only inside training folds;
- naive, linear, tree, ensemble, and kernel baselines are compared;
- accuracy is supplemented by macro F1, class-level metrics, and confusion analysis;
- model improvement is interpreted together with overfitting and representation limits.

This gap is narrower than claiming a novel state-of-the-art MGR algorithm. It is nevertheless non-trivial because the literature shows that uncontrolled evaluation can produce misleading conclusions. A carefully executed classical comparison contributes evidence about representation sufficiency, non-linear decision boundaries, and reproducibility under limited data.

> **Research gap:** Existing results often combine different representations, classifiers, datasets, and evaluation protocols. This makes it difficult to isolate whether reported improvements arise from the model itself. The thesis addresses this problem through a controlled and reproducible comparison of multiple classifiers using the same engineered feature representation and evaluation protocol.

---

## E. Positioning of the Thesis Contribution

The thesis differs from work focused on maximum benchmark accuracy. Its contribution is a controlled engineering comparison of classical models on one explicitly defined feature representation. The main comparative dimension is classifier behavior under constant data and preprocessing conditions.

The justified baseline set is:

- dummy classifier, establishing the chance-level lower bound;
- logistic regression, testing approximate linear separability;
- decision tree, providing a simple non-linear reference;
- random forest, representing a strong bagged tree ensemble;
- gradient boosting, representing a sequential ensemble;
- RBF SVM, testing a flexible margin-based decision boundary.

The literature supports RBF SVM because non-linear kernel methods have repeatedly been effective on aggregated audio descriptors [3]. Random forest is justified as a strong tabular baseline that captures interactions without scale sensitivity. Logistic regression is necessary to determine whether the extracted representation is already close to linearly separable. The dummy model verifies that performance is not an artifact of class distribution.

The thesis does not claim that global features and RBF SVM are superior to convolutional or transfer-learning systems. Instead, it investigates whether the additional complexity of a kernel classifier is justified relative to conventional baselines on a small dataset. It also identifies the point at which further improvement likely requires richer temporal representation or additional independent data rather than more aggressive tuning of the same feature table.

This positioning leads directly to the experimental design:

1. use the unscaled extracted feature table as the common input;
2. remove exact duplicates before the split;
3. create a fixed stratified training and test partition;
4. fit scaling and optional feature selection inside cross-validation;
5. compare all models with the same primary and supporting metrics;
6. preserve the test set for final evaluation;
7. analyze class-specific errors, stability, and the train-validation gap;
8. limit conclusions to the dataset and protocol actually evaluated.

The resulting study is scientifically grounded because it follows established MGR methodology, responds directly to documented evaluation weaknesses, and defines a contribution appropriate for an engineering thesis.

---

## Preliminary Bibliography

[1] G. Tzanetakis and P. Cook, "Musical Genre Classification of Audio Signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293-302, 2002, doi: [10.1109/TSA.2002.800560](https://doi.org/10.1109/TSA.2002.800560).

[2] J. Bergstra, N. Casagrande, D. Erhan, D. Eck, and B. Kegl, "Aggregate Features and AdaBoost for Music Classification," *Machine Learning*, vol. 65, pp. 473-484, 2006, doi: [10.1007/s10994-006-9019-7](https://doi.org/10.1007/s10994-006-9019-7).

[3] I. Panagakis, E. Benetos, and C. Kotropoulos, "Music Genre Classification: A Multilinear Approach," in *Proceedings of the 9th International Conference on Music Information Retrieval (ISMIR)*, 2008, pp. 583-588. [Online]. Available: [ISMIR PDF](https://archives.ismir.net/ismir2008/paper/000181.pdf).

[4] K. Choi, G. Fazekas, M. Sandler, and K. Cho, "Transfer Learning for Music Classification and Regression Tasks," in *Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)*, 2017, pp. 141-149. [Online]. Available: [ISMIR PDF](https://archives.ismir.net/ismir2017/paper/000012.pdf).

[5] B. L. Sturm, "Classification Accuracy Is Not Enough: On the Evaluation of Music Genre Recognition Systems," *Journal of Intelligent Information Systems*, vol. 41, pp. 371-406, 2013, doi: [10.1007/s10844-013-0250-y](https://doi.org/10.1007/s10844-013-0250-y).

[6] F. Rodriguez-Algarra, B. L. Sturm, and S. Dixon, "Characterising Confounding Effects in Music Classification Experiments through Interventions," *Transactions of the International Society for Music Information Retrieval*, vol. 2, no. 1, pp. 52-66, 2019, doi: [10.5334/tismir.24](https://doi.org/10.5334/tismir.24).

[7] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, "FMA: A Dataset for Music Analysis," in *Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)*, 2017, pp. 316-323. [Online]. Available: [ISMIR PDF](https://archives.ismir.net/ismir2017/paper/000075.pdf).

[8] R. Mignot and G. Peeters, "An Analysis of the Effect of Data Augmentation Methods: Experiments for a Musical Genre Classification Task," *Transactions of the International Society for Music Information Retrieval*, vol. 2, no. 1, pp. 97-110, 2019, doi: [10.5334/tismir.26](https://doi.org/10.5334/tismir.26).

[9] B. L. Sturm, "The GTZAN Dataset: Its Contents, Its Faults, Their Effects on Evaluation, and Its Future Use," arXiv:1306.1461, 2013. [Online]. Available: [arXiv](https://arxiv.org/abs/1306.1461).

[10] Music Information Retrieval Evaluation eXchange, "MIREX Audio Genre Classification and Audio Classification Train/Test Tasks," 2005-2017. [Online]. Available: [MIREX 2005 results](https://music-ir.org/mirex/wiki/2005%3AAudio_Genre_Classification_Results) and [MIREX 2017 protocol](https://music-ir.org/mirex/wiki/2017%3AAudio_Classification_%28Train/Test%29_Tasks).

[11] M. Defferrard, S. P. Mohanty, S. F. Carroll, and M. Salathe, "Learning to Recognize Musical Genre from Audio: Challenge Overview," in *Companion Proceedings of The Web Conference 2018*, pp. 1921-1922, 2018, doi: [10.1145/3184558.3192310](https://doi.org/10.1145/3184558.3192310).

[12] D. Bogdanov, A. Porter, J. Urbano, and H. Schreiber, "The MediaEval 2017 AcousticBrainz Genre Task: Content-Based Music Genre Recognition from Multiple Sources," in *Working Notes Proceedings of the MediaEval 2017 Workshop*, CEUR-WS, vol. 1984, 2017. [Online]. Available: [CEUR-WS PDF](https://ceur-ws.org/Vol-1984/Mediaeval_2017_paper_6.pdf).
