#!/usr/bin/env python3
"""Longitudinal lexical-development analysis for PELIC.

The script never writes learner text to results. Output tables contain identifiers,
metadata, aggregate metrics, and model estimates only. Because PELIC is licensed
CC BY-NC-ND 4.0, do not redistribute processed text or token sequences.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_raw" / "PELIC-dataset"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

METRICS = [
    "mattr50",
    "mtld",
    "lexical_density",
    "mean_log_frequency_content",
    "rare_content_rate",
]

CONTENT_PREFIXES = ("NN", "VB", "JJ", "RB")
WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


def load_frequency_reference() -> dict[str, float]:
    freq = pd.read_csv(DATA / "corpus_stats" / "lemma_frequencies.csv")
    return dict(zip(freq["lemma"].astype(str).str.lower(), freq["per_M"].astype(float)))


def mark_near_duplicates(df: pd.DataFrame, normalized: pd.Series, threshold: float = 0.90) -> pd.Series:
    """Greedy within-learner 5-gram Jaccard audit in chronological row order."""
    flags = pd.Series(False, index=df.index)
    shingle_cache: dict[int, set[tuple[str, ...]]] = {}
    for idx, value in normalized.items():
        words = value.split()
        shingle_cache[idx] = set(zip(*(words[i:] for i in range(5)))) if len(words) >= 5 else {tuple(words)}
    for _, group in df.groupby("anon_id", sort=False):
        kept: list[int] = []
        for idx in group.index:
            candidate = shingle_cache[idx]
            duplicate = False
            for prior in kept:
                reference = shingle_cache[prior]
                union = len(candidate | reference)
                similarity = len(candidate & reference) / union if union else 1.0
                if similarity >= threshold:
                    duplicate = True
                    break
            flags.loc[idx] = duplicate
            if not duplicate:
                kept.append(idx)
    return flags


def mattr(tokens: list[str], window: int = 50) -> float:
    if not tokens:
        return float("nan")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([
        len(set(tokens[i : i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]))


def _mtld_one_direction(tokens: list[str], threshold: float = 0.72) -> float:
    if not tokens:
        return float("nan")
    factors = 0.0
    types: set[str] = set()
    count = 0
    for token in tokens:
        count += 1
        types.add(token)
        ttr = len(types) / count
        if ttr <= threshold:
            factors += 1.0
            types = set()
            count = 0
    if count:
        ttr = len(types) / count
        factors += (1.0 - ttr) / (1.0 - threshold)
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def mtld(tokens: list[str]) -> float:
    return float(np.mean([
        _mtld_one_direction(tokens),
        _mtld_one_direction(list(reversed(tokens))),
    ]))


def parse_metrics(value: str, frequency_reference: dict[str, float]) -> dict[str, float]:
    triples = ast.literal_eval(value)
    words: list[str] = []
    content: list[str] = []
    for token, lemma, pos in triples:
        form = str(lemma).lower().strip()
        if not WORD_RE.match(form) or form.startswith("anon_"):
            continue
        words.append(form)
        if str(pos).startswith(CONTENT_PREFIXES):
            content.append(form)
    per_million = np.array([frequency_reference.get(w, 0.0) for w in content], dtype=float)
    log_frequency = np.log10(per_million + 0.1)
    return {
        "n_lexical_tokens": len(words),
        "n_content_tokens": len(content),
        "mattr50": mattr(words, 50),
        "mtld": mtld(words),
        "lexical_density": len(content) / len(words) if words else float("nan"),
        "mean_log_frequency_content": float(log_frequency.mean()) if len(log_frequency) else float("nan"),
        "rare_content_rate": float((per_million < 10.0).mean()) if len(per_million) else float("nan"),
    }


def load_sample() -> pd.DataFrame:
    usecols = [
        "answer_id", "anon_id", "L1", "gender", "semester", "level_id",
        "class_id", "question_id", "version", "text_len", "text", "tok_lem_POS",
    ]
    df = pd.read_csv(DATA / "PELIC_compiled.csv", usecols=usecols, low_memory=False)
    questions = pd.read_csv(
        DATA / "corpus_files" / "question.csv",
        usecols=["question_id", "question_type_id"],
    )
    df = df.merge(questions, on="question_id", how="left", validate="many_to_one")
    df = df[
        (df["class_id"] == "w")
        & (df["version"] == 1)
        & (df["text_len"] >= 100)
        & (df["level_id"].isin([3, 4, 5]))
    ].copy()
    nlevels = df.groupby("anon_id")["level_id"].nunique()
    df = df[df["anon_id"].isin(nlevels[nlevels >= 2].index)].copy()
    df["question_type"] = df["question_type_id"].fillna(-1).astype(int).astype(str)
    major_l1 = df.drop_duplicates("anon_id")["L1"].value_counts()
    keep_l1 = set(major_l1[major_l1 >= 10].index)
    df["L1_group"] = df["L1"].where(df["L1"].isin(keep_l1), "Other")
    normalized = df["text"].fillna("").str.lower().str.replace(r"[^a-z]+", " ", regex=True).str.strip()
    df["normalized_text_hash"] = normalized.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    df["near_duplicate_5gram_jaccard90"] = mark_near_duplicates(df, normalized, 0.90)
    return df.sort_values(["anon_id", "semester", "answer_id"]).reset_index(drop=True)


def fit_hybrid(data: pd.DataFrame, metric: str, label: str) -> dict[str, object]:
    d = data.dropna(subset=[metric]).copy()
    d["outcome_z"] = (d[metric] - d[metric].mean()) / d[metric].std(ddof=0)
    d["student_mean_level"] = d.groupby("anon_id")["level_id"].transform("mean")
    d["level_within"] = d["level_id"] - d["student_mean_level"]
    d["level_between"] = d["student_mean_level"] - d["student_mean_level"].mean()
    d["log_len_z"] = (np.log(d["text_len"]) - np.log(d["text_len"]).mean()) / np.log(d["text_len"]).std(ddof=0)
    formula = "outcome_z ~ level_within + level_between + log_len_z + C(question_type) + C(L1_group)"
    model = smf.mixedlm(formula, d, groups=d["anon_id"], re_formula="1")
    try:
        fit = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        if not fit.converged:
            fit = model.fit(reml=False, method="powell", maxiter=2000, disp=False)
    except Exception:
        fit = model.fit(reml=False, method="powell", maxiter=2000, disp=False)
    rows = []
    for term in ["level_within", "level_between", "log_len_z"]:
        estimate = float(fit.params[term])
        se = float(fit.bse[term])
        z = estimate / se
        rows.append({
            "analysis": label,
            "metric": metric,
            "term": term,
            "estimate_std": estimate,
            "se": se,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
            "z": z,
            "p_value": 2 * norm.sf(abs(z)),
            "n_texts": len(d),
            "n_students": d["anon_id"].nunique(),
            "converged": bool(fit.converged),
            "aic": float(fit.aic),
        })
    return {"rows": rows, "fit": fit}


def fit_l1_slopes(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    output = []
    for l1 in ["Arabic", "Chinese", "Korean", "Japanese"]:
        d = data[data["L1"] == l1].dropna(subset=[metric]).copy()
        eligible = d.groupby("anon_id")["level_id"].nunique()
        d = d[d["anon_id"].isin(eligible[eligible >= 2].index)].copy()
        if d["anon_id"].nunique() < 10:
            continue
        d["outcome_z"] = (d[metric] - d[metric].mean()) / d[metric].std(ddof=0)
        d["mean_level"] = d.groupby("anon_id")["level_id"].transform("mean")
        d["level_within"] = d["level_id"] - d["mean_level"]
        d["log_len_z"] = (np.log(d["text_len"]) - np.log(d["text_len"]).mean()) / np.log(d["text_len"]).std(ddof=0)
        # Student fixed effects isolate within-learner change in each L1 subgroup;
        # cluster-robust SEs retain the learner as the independence unit.
        fit = smf.ols(
            "outcome_z ~ level_within + log_len_z + C(question_type) + C(anon_id)",
            d,
        ).fit(cov_type="cluster", cov_kwds={"groups": d["anon_id"]})
        est, se = float(fit.params["level_within"]), float(fit.bse["level_within"])
        output.append({
            "metric": metric, "L1": l1, "estimate_std": est,
            "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
            "p_value": 2 * norm.sf(abs(est / se)), "n_texts": len(d),
            "n_students": d["anon_id"].nunique(),
        })
    return pd.DataFrame(output)


def make_forest(results: pd.DataFrame) -> None:
    d = results[(results["analysis"] == "primary") & (results["term"].isin(["level_within", "level_between"]))].copy()
    labels = {
        "mattr50": "MATTR-50", "mtld": "MTLD",
        "lexical_density": "Lexical density",
        "mean_log_frequency_content": "Mean corpus log frequency",
        "rare_content_rate": "Rare content-lemma rate",
    }
    fig, ax = plt.subplots(figsize=(8, 5.2))
    positions = np.arange(len(METRICS))
    for offset, term, color, marker in [(-0.12, "level_within", "#1f77b4", "o"), (0.12, "level_between", "#d62728", "s")]:
        q = d[d["term"] == term].set_index("metric").loc[METRICS]
        y = positions + offset
        ax.errorbar(q["estimate_std"], y,
                    xerr=[q["estimate_std"] - q["ci_low"], q["ci_high"] - q["estimate_std"]],
                    fmt=marker, color=color, capsize=3, label=term.replace("level_", "").title())
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions, [labels[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized change per course level (95% CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    # High-resolution source suitable for full-width journal production.
    fig.savefig(FIGURES / "within_between_forest.png", dpi=600)
    plt.close(fig)


def paired_level_changes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        means = data.groupby(["anon_id", "level_id"])[metric].mean().unstack()
        overall_sd = data[metric].std(ddof=0)
        for low, high in [(3, 4), (4, 5), (3, 5)]:
            if low not in means or high not in means:
                continue
            pair = means[[low, high]].dropna()
            difference = pair[high] - pair[low]
            t_result = ttest_rel(pair[high], pair[low])
            try:
                w_p = float(wilcoxon(difference).pvalue)
            except ValueError:
                w_p = float("nan")
            se = difference.std(ddof=1) / math.sqrt(len(difference))
            rows.append({
                "metric": metric, "transition": f"{low}_to_{high}",
                "n_students": len(difference), "mean_low": pair[low].mean(),
                "mean_high": pair[high].mean(), "mean_change": difference.mean(),
                "standardized_change": difference.mean() / overall_sd,
                "ci_low": difference.mean() - 1.96 * se,
                "ci_high": difference.mean() + 1.96 * se,
                "paired_t_p": float(t_result.pvalue), "wilcoxon_p": w_p,
            })
    out = pd.DataFrame(rows)
    out["paired_t_fdr"] = multipletests(out["paired_t_p"], method="fdr_bh")[1]
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    frequency_reference = load_frequency_reference()
    metric_rows = [parse_metrics(v, frequency_reference) for v in df["tok_lem_POS"]]
    metrics = pd.DataFrame(metric_rows)
    analysis = pd.concat([df.drop(columns=["text", "tok_lem_POS"]).reset_index(drop=True), metrics], axis=1)
    analysis.to_csv(RESULTS / "analysis_metrics.csv", index=False)

    sample = {
        "n_texts": int(len(analysis)),
        "n_students": int(analysis["anon_id"].nunique()),
        "n_questions": int(analysis["question_id"].nunique()),
        "level_counts": {str(k): int(v) for k, v in analysis["level_id"].value_counts().sort_index().items()},
        "question_type_counts": {str(k): int(v) for k, v in analysis["question_type"].value_counts().sort_index().items()},
        "l1_student_counts": {str(k): int(v) for k, v in analysis.drop_duplicates("anon_id")["L1"].value_counts().items()},
        "exact_duplicate_rows_within_learner": int(analysis.duplicated(["anon_id", "normalized_text_hash"]).sum()),
        "near_duplicate_rows_within_learner": int(analysis["near_duplicate_5gram_jaccard90"].sum()),
    }
    (RESULTS / "sample_summary.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")

    descriptive = analysis.groupby("level_id")[METRICS + ["text_len"]].agg(["count", "mean", "std", "median"])
    descriptive.columns = ["_".join(c) for c in descriptive.columns]
    descriptive.reset_index().to_csv(RESULTS / "descriptive_by_level.csv", index=False)

    rows = []
    for metric in METRICS:
        rows.extend(fit_hybrid(analysis, metric, "primary")["rows"])
        paragraph = analysis[analysis["question_type"] == "1"]
        eligible = paragraph.groupby("anon_id")["level_id"].nunique()
        paragraph = paragraph[paragraph["anon_id"].isin(eligible[eligible >= 2].index)]
        rows.extend(fit_hybrid(paragraph, metric, "paragraph_only")["rows"])
        min150 = analysis[analysis["text_len"] >= 150]
        eligible150 = min150.groupby("anon_id")["level_id"].nunique()
        min150 = min150[min150["anon_id"].isin(eligible150[eligible150 >= 2].index)]
        rows.extend(fit_hybrid(min150, metric, "min150_words")["rows"])
        dedup = analysis.drop_duplicates(["anon_id", "normalized_text_hash"], keep="first")
        eligible_dedup = dedup.groupby("anon_id")["level_id"].nunique()
        dedup = dedup[dedup["anon_id"].isin(eligible_dedup[eligible_dedup >= 2].index)]
        rows.extend(fit_hybrid(dedup, metric, "exact_deduplicated")["rows"])
        near_dedup = analysis[~analysis["near_duplicate_5gram_jaccard90"]]
        eligible_near = near_dedup.groupby("anon_id")["level_id"].nunique()
        near_dedup = near_dedup[near_dedup["anon_id"].isin(eligible_near[eligible_near >= 2].index)]
        rows.extend(fit_hybrid(near_dedup, metric, "near_deduplicated")["rows"])
    model_results = pd.DataFrame(rows)
    for analysis_name in model_results["analysis"].unique():
        mask = (model_results["analysis"] == analysis_name) & (model_results["term"] == "level_within")
        model_results.loc[mask, "p_fdr"] = multipletests(model_results.loc[mask, "p_value"], method="fdr_bh")[1]
    model_results.to_csv(RESULTS / "hybrid_model_results.csv", index=False)

    l1_results = pd.concat([fit_l1_slopes(analysis, metric) for metric in METRICS], ignore_index=True)
    l1_results.to_csv(RESULTS / "l1_within_slopes.csv", index=False)
    paired_level_changes(analysis).to_csv(RESULTS / "paired_level_changes.csv", index=False)
    make_forest(model_results)

    dataset_commit = subprocess.check_output(
        ["git", "-C", str(DATA), "rev-parse", "HEAD"], text=True
    ).strip()
    run_info = {
        "dataset_commit": dataset_commit,
        "eligibility": "writing; version 1; text_len >=100; levels 3-5; learner observed at >=2 levels",
        "metrics": METRICS,
        "rarity_reference": "PELIC lemma frequency table",
        "rare_threshold_per_million": 10.0,
        "mattr_window": 50,
        "mtld_threshold": 0.72,
    }
    (RESULTS / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(sample, indent=2))
    print(model_results[model_results["term"].isin(["level_within", "level_between"])].to_string(index=False))


if __name__ == "__main__":
    main()
