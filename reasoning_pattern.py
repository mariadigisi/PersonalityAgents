"""
Stimulus - Cognition - Click analysis
=====================================

Topic modelling (BERTopic) of the free-text reasoning written by the victim
agents before each CLICK / NO_CLICK decision, followed by the statistical
analysis reported in Section 4.3 of the paper ("Cognitive Reasoning Pattern
Analysis"):

  1. BERTopic clustering of the reasoning texts into cognitive patterns
  2. pattern x decision           (chi-square, Cramer's V, standardized residuals)
  3. pattern x MBTI dimension     (E/I, S/N, T/F, J/P)
  4. pattern x persuasion lever   (overall and lever-stratified)
  5. heatmap figure (pattern by MBTI profile, pattern by decision)

Input: one or more `<run>_phishing_report.json` files produced by the phishing
simulation. Each file must contain a "responses" list whose items include at
least the fields: "thinking", "decision", "mbti", "lever".

Note: the agents' reasoning texts are in Italian, so the c-TF-IDF vectorizer
uses Italian stop words and a multilingual sentence-embedding model.

Usage
-----
    pip install bertopic sentence-transformers spacy seaborn scipy
    python stimulus_cognition_click.py --runs 1 2 3 4 --data-dir . --out-dir results
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.it.stop_words import STOP_WORDS as ITALIAN_STOP_WORDS
from umap import UMAP

# ---------------------------------------------------------------------------
# Configuration (identical to the settings used for the paper)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# HDBSCAN initially finds more topics (11 in the reported run); BERTopic then
# merges them down to NR_TOPICS, which includes the outlier topic (-1).
# NR_TOPICS = 6 therefore yields 5 cognitive patterns + outliers.
NR_TOPICS = 6

# Domain-specific words removed from the c-TF-IDF representation, because they
# appear in almost every text and do not help distinguish the patterns.
EXTRA_STOP_WORDS = [
    "security", "email", "click", "link", "company", "corporate",
    "this", "all", "these", "account", "warning", "urgent", "ceo",
    "click", "clicks", "clicked", "act", "action", "I will act", "it",
]

# Human-readable names assigned manually after reading the texts of each
# cluster. Topic ids are only guaranteed to match this mapping for the run
# reported in the paper; check the printed keywords before relying on it.
PATTERN_NAMES = {
    -1: "Outliers",
    0: "Technical recognition",
    1: "Interception anxiety",
    2: "Reward-driven",
    3: "Loss/urgency pressure",
    4: "Procedural compliance",
}

MBTI_DIMENSIONS = {"E/I": 0, "S/N": 1, "T/F": 2, "J/P": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def chi_square(table, label):
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"{label}: chi2 = {chi2:.2f}, p = {p:.4f}, dof = {dof}, n = {int(table.values.sum())}")
    return chi2, p, dof, expected


def cramers_v(chi2, table):
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def standardized_residuals(table, expected):
    return (table - expected) / np.sqrt(expected)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def load_reports(data_dir, runs):
    frames = []
    for run in runs:
        path = os.path.join(data_dir, f"{run}_phishing_report.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frame = pd.DataFrame(data["responses"])
        frame["run"] = run
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(runs)} run(s): {df.shape[0]} responses, {df.shape[1]} columns")
    return df


def build_topic_model():
    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5,
                            metric="euclidean", cluster_selection_method="eom")
    stop_words = list(ITALIAN_STOP_WORDS) + EXTRA_STOP_WORDS
    vectorizer_model = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
    return BERTopic(embedding_model=EMBEDDING_MODEL, umap_model=umap_model,
                    hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
                    nr_topics=NR_TOPICS, verbose=True)


def describe_topics(topic_model, topics):
    section("1. Topics found by BERTopic (c-TF-IDF keywords)")
    print(topic_model.get_topic_info()[["Topic", "Count", "Name"]].to_string(index=False))
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)[:5]]
        name = PATTERN_NAMES.get(topic_id, "?")
        print(f"\nTopic {topic_id} ({name}) keywords: {', '.join(words)}")
        for doc in topic_model.get_representative_docs(topic_id)[:2]:
            print(f"   e.g.: {doc[:200]}...")


def pattern_vs_decision(df_clean):
    section("2. Cognitive pattern x decision (outliers excluded)")
    counts = pd.crosstab(df_clean["topic"], df_clean["decision"])
    rates = pd.crosstab(df_clean["topic"], df_clean["decision"], normalize="index") * 100
    print("Counts:\n", counts, "\n\nClick rate by pattern (%):\n", rates.round(2))
    chi2, _, _, expected = chi_square(counts, "Pattern x decision")
    print(f"Cramer's V = {cramers_v(chi2, counts):.4f}")
    print("\nStandardized residuals:\n", standardized_residuals(counts, expected).round(2))


def pattern_vs_mbti(df_clean):
    section("3. Cognitive pattern x MBTI")
    profile_dist = pd.crosstab(df_clean["mbti"], df_clean["topic"], normalize="index") * 100
    print("Pattern distribution within each MBTI profile (%):\n", profile_dist.round(1))
    print()
    for dim in MBTI_DIMENSIONS:
        chi_square(pd.crosstab(df_clean["topic"], df_clean[dim]), f"Pattern x {dim}")

    print("\nJ/P composition of the two most click-prone patterns:")
    for topic_id in (4, 3):
        sub = df_clean[df_clean["topic"] == topic_id]
        comp = sub["J/P"].value_counts()
        total = len(sub)
        shares = ", ".join(f"{k} {v}/{total} ({100 * v / total:.1f}%)" for k, v in comp.items())
        print(f"  Topic {topic_id} ({PATTERN_NAMES.get(topic_id, '?')}): {shares}")


def pattern_vs_lever(df, df_clean):
    section("4. Cognitive pattern x persuasion lever")
    chi_square(pd.crosstab(df["topic"], df["lever"]), "Pattern x lever (outliers included)")
    print("\nCounts (outliers excluded):\n", pd.crosstab(df_clean["topic"], df_clean["lever"]))

    print("\nLever-stratified analysis: pattern x decision within each lever")
    significant = []
    for lever in df_clean["lever"].unique():
        sub = df_clean[df_clean["lever"] == lever]
        table = pd.crosstab(sub["topic"], sub["decision"])
        if table.shape[0] > 1 and table.shape[1] > 1:
            _, p, _, _ = chi_square(table, f"  {lever}")
            if p < 0.05:
                significant.append(lever)

    for lever in significant:
        sub = df_clean[df_clean["lever"] == lever]
        table = pd.crosstab(sub["topic"], sub["decision"])
        _, _, _, expected = chi2_contingency(table)
        print(f"\n--- {lever} ---\n", table)
        print("Standardized residuals:\n", standardized_residuals(table, expected).round(2))
        print("MBTI profiles in each pattern:")
        for topic_id in sorted(sub["topic"].unique()):
            profiles = sub[sub["topic"] == topic_id]["mbti"].value_counts()
            print(f"  Topic {topic_id} ({PATTERN_NAMES.get(topic_id, '?')}): {profiles.to_dict()}")


def plot_heatmaps(df_clean, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_plot = df_clean.copy()
    order = [PATTERN_NAMES.get(t, f"Topic {t}") for t in sorted(df_plot["topic"].unique())]
    df_plot["pattern"] = df_plot["topic"].map(lambda t: PATTERN_NAMES.get(t, f"Topic {t}"))

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "figure.titlesize": 13})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.3, 1]})

    by_profile = pd.crosstab(df_plot["mbti"], df_plot["pattern"], normalize="index") * 100
    by_profile = by_profile.reindex(columns=order, fill_value=0)
    sns.heatmap(by_profile, annot=True, fmt=".1f", cmap="Blues", linewidths=.3,
                cbar_kws={"label": "% of reasoning texts in the profile"}, ax=ax1)
    ax1.set_title("A. Cognitive patterns by MBTI profile", pad=15, weight="bold")
    ax1.set_xlabel("Cognitive pattern", labelpad=10)
    ax1.set_ylabel("Agent MBTI profile", labelpad=10)
    plt.setp(ax1.get_xticklabels(), rotation=35, ha="right")

    by_decision = pd.crosstab(df_plot["pattern"], df_plot["decision"], normalize="columns") * 100
    by_decision = by_decision.reindex(index=order, fill_value=0)
    sns.heatmap(by_decision, annot=True, fmt=".1f", cmap="Reds", linewidths=.3,
                cbar_kws={"label": "% of all decisions of that type"}, ax=ax2)
    ax2.set_title("B. Cognitive patterns by final decision", pad=15, weight="bold")
    ax2.set_xlabel("Agent decision", labelpad=10)
    ax2.set_ylabel("")
    plt.setp(ax2.get_yticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BERTopic analysis of victim-agent reasoning.")
    parser.add_argument("--data-dir", default=".", help="folder containing <run>_phishing_report.json")
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="run numbers to include (default: 1 2 3 4, as in the reported analysis)")
    parser.add_argument("--out-dir", default="results", help="output folder")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_reports(args.data_dir, args.runs)
    docs = df["thinking"].astype(str).tolist()

    topic_model = build_topic_model()
    topics, _ = topic_model.fit_transform(docs)
    df["topic"] = topics
    df["pattern"] = df["topic"].map(lambda t: PATTERN_NAMES.get(t, f"Topic {t}"))
    for dim, pos in MBTI_DIMENSIONS.items():
        df[dim] = df["mbti"].str[pos]

    df_clean = df[df["topic"] != -1].copy()
    print(f"\nReasoning texts: {len(df)} | assigned to a pattern: {len(df_clean)} "
          f"| outliers: {len(df) - len(df_clean)}")

    describe_topics(topic_model, topics)
    pattern_vs_decision(df_clean)
    pattern_vs_mbti(df_clean)
    pattern_vs_lever(df, df_clean)
    plot_heatmaps(df_clean, os.path.join(args.out_dir, "cognitive_patterns_heatmap.pdf"))

    out_csv = os.path.join(args.out_dir, "phishing_with_topics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Annotated data saved to {out_csv}")


if __name__ == "__main__":
    main()
