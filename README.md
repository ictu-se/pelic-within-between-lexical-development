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
