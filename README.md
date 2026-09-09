# Within- and between-learner lexical development in PELIC

This repository reproduces the aggregate analyses for a longitudinal study of
lexical development in University of Pittsburgh English Language Institute
Corpus (PELIC) writing. It estimates within-learner and between-learner
course-level associations for MATTR-50, MTLD, lexical density, and two
corpus-relative frequency measures.

## Data

PELIC v1.1 is available from Zenodo at DOI `10.5281/zenodo.3991977` and from
the official PELIC repository. Clone the dataset into
`dataset_raw/PELIC-dataset` and check out commit
`c4526baeb8fb5d69732f9e2a8e1430b41ed38c53`. The corpus is licensed CC
BY-NC-ND 4.0. This repository does not redistribute learner texts, transformed
texts, token sequences, or row-level derived data.

```bash
git clone https://github.com/ELI-Data-Mining-Group/PELIC-dataset.git dataset_raw/PELIC-dataset
git -C dataset_raw/PELIC-dataset checkout c4526baeb8fb5d69732f9e2a8e1430b41ed38c53
git -C dataset_raw/PELIC-dataset lfs pull
```

## Reproduce the analysis

Python 3.9 was used for the reported run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/run_experiment.py
```

The script applies the documented eligibility rules, calculates all five
outcomes, fits the primary and sensitivity models, performs paired transition
and L1 analyses, and recreates the aggregate tables and forest plot under
`results/`. Package versions are pinned in `requirements.txt`; the dataset
commit and analysis constants are recorded in the generated run metadata.

## Sharing constraints

Only code, documentation, figures, and aggregate output tables should be made
public. Do not commit the dataset directory or `results/analysis_metrics.csv`.
Users must obtain PELIC from its official source and comply with its licence.

## Direct within-between comparison

The revised analysis is titled **Separating within-learner lexical change from
between-learner course-level differences in L2 writing**. To reproduce the added
post hoc contrasts and both manuscript figures, first run the primary experiment
above to create the local row-level metrics, then run:

```sh
python3 src/compare_associations.py
python3 src/plot_associations.py
```

The comparison script refits each primary model, checks convergence and agreement
with the stored primary coefficients, and computes between-minus-within contrasts
using their joint covariance. A separate Benjamini-Hochberg adjustment covers the
five contrast tests. The aggregate output is `results/within_between_contrasts.csv`;
figures are in `results/figures/`.

MATTR and MTLD show positive within-learner associations. Their between-minus-within
contrasts have adjusted p values of approximately .149 and .116, respectively.
Neither meets the five-test FDR threshold; larger between-learner point estimates
should not be described as an established difference.

The experiment also includes chronology, learner-question clustering, and
leave-one-text-out frequency checks. The historical `question_random_intercept`
label in aggregate outputs denotes a variance component nested within learners,
not a crossed prompt intercept shared across learners. No question identifier
spans levels, so prompt-set differences remain confounded with level.

Only aggregate results and reproducible research code are published here.
Manuscript files, editorial correspondence, licensed corpus texts and row-level
derived data are excluded. See `REPRODUCIBILITY.md` for the release inventory.
