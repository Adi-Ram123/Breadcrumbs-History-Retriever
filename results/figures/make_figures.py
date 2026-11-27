"""Figures and per-condition confidence intervals for report.md.

Reads only saved results (no model runs):
  results/test/answer_ranks.parquet, results/test/test_scores.csv, results/test/rerank_k.csv,
  results/experiments/*/validation.json.
Writes results/figures/*.png and results/figures/condition_ci.csv.

usage: uv run --no-project --with pandas --with pyarrow --with matplotlib python results/figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures"
EXP = ROOT / "results" / "experiments"

# Validated categorical order (blue, orange); light blue is a lighter step of the same hue; gray for baselines.
BLUE, ORANGE, LIGHT = "#2a78d6", "#eb6834", "#86b6ef"
GRAY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})

# Test conditions in the order of the approach (results/test numbering).
STEPS = [(1, "1. Keyword search\n(BM25)"), (3, "2. Embedding model,\noff the shelf"),
         (4, "3. Embedding model,\nfine-tuned"), (7, "4. + fine-tuned\nreranker (final)")]


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def condition_ci(n_boot=10_000, seed=0):
    """Recall@1 per condition and question writer, averaged over seeds per question, with a 95% bootstrap CI."""
    r = pd.read_parquet(ROOT / "results/test/answer_ranks.parquet")
    r["hit"] = (r["rank"] == 1).astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for (src, cond), g in r.groupby(["source", "condition"]):
        per_q = g.groupby("qid")["hit"].mean().to_numpy()
        boot = per_q[rng.integers(0, len(per_q), size=(n_boot, len(per_q)))].mean(axis=1)
        rows.append({"source": src, "condition": cond, "n": len(per_q), "seeds": g["seed"].nunique(),
                     "recall@1": per_q.mean(), "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "condition_ci.csv", index=False)
    return df


def fig_steps(ci):
    d = ci[ci.source == "llama"].set_index("condition")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(STEPS))
    vals = [d.loc[c, "recall@1"] for c, _ in STEPS]
    lo = [d.loc[c, "recall@1"] - d.loc[c, "ci_low"] for c, _ in STEPS]
    hi = [d.loc[c, "ci_high"] - d.loc[c, "recall@1"] for c, _ in STEPS]
    ax.bar(x, vals, 0.6, color=[GRAY, LIGHT, LIGHT, BLUE])
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
    for xi, v, h in zip(x, vals, hi):
        ax.text(xi, v + h + 0.012, f"{v:.3f}", ha="center", fontsize=11, color=INK)
    ax.set_xticks(x, [s for _, s in STEPS])
    ax.set_ylim(0.4, 0.92)
    ax.set_ylabel("Recall@1, 2,000 test questions")
    ax.grid(axis="x", visible=False)
    ax.set_title("Each step of the approach puts the right page first more often", loc="left", color=INK, fontsize=12)
    save(fig, "fig1_approach_steps.png")


def fig_recall_at_k():
    s = pd.read_csv(ROOT / "results/test/test_scores.csv")
    s = s[s.source == "llama"].groupby("condition").mean(numeric_only=True)
    ks = [1, 5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    for c, color, label in [(1, GRAY, "keyword search (BM25)"), (3, LIGHT, "embedding model, off the shelf"),
                            (4, BLUE, "embedding model, fine-tuned")]:
        ax.plot(ks, [s.loc[c, f"recall@{k}"] for k in ks], color=color, lw=2, marker="o", ms=6, label=label)
    ax.annotate("in the top 10: 98.0%", (10, s.loc[4, "recall@10"]), textcoords="offset points", xytext=(8, -22),
                fontsize=9.5, color=INK)
    ax.annotate("ranked first: 80.6%", (1, s.loc[4, "recall@1"]), textcoords="offset points", xytext=(8, -4),
                fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k, test")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("The fine-tuned model finds the page but often not first", loc="left", color=INK, fontsize=12)
    save(fig, "fig2_recall_at_k.png")


def fig_rerank_k():
    k = pd.read_csv(ROOT / "results/test/rerank_k.csv")
    k = k[(k.condition == 7) & (k.kind == "fused")].groupby("k")["recall@1"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(k.index, k.values, color=BLUE, lw=2, marker="o", ms=6)
    ax.axhline(0.806, color=GRAY, lw=1.5, ls="--")
    ax.text(50, 0.8065, "fine-tuned embedding model alone", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(0.795, 0.84)
    ax.set_xlabel("k (candidates the reranker reorders)")
    ax.set_ylabel("Recall@1, test")
    ax.set_title("Reranking more than the top 10 adds almost nothing", loc="left", color=INK, fontsize=11)
    save(fig, "fig3_rerank_k.png")


def fig_finetune_each_model():
    emb = [("snowflake-arctic-\nembed-l-v2.0 (chosen)", "first-stage-snowflake", "snowflake-arctic-embed-l-v2.0", "exp_first-stage-snowflake"),
           ("mxbai-embed-\nlarge-v1", "first-stage-mxbai", "mxbai-embed-large-v1", "exp_first-stage-mxbai"),
           ("e5-large-v2", "first-stage-e5", "e5-large-v2", "exp_first-stage-e5")]
    rer = [("gte-reranker-\nmodernbert-base (chosen)", "ft-gte-reranker-modernbert-base-on-snowflake", "pretrained_k10", "reranker_k10"),
           ("bge-reranker-\nv2-m3", "ft-bge-reranker-v2-m3-on-snowflake", "pretrained_k10", "reranker_k10")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [3, 2]}, sharey=True)
    for ax, rows, title in [(axes[0], emb, "Embedding models (step 3)"), (axes[1], rer, "Rerankers on the top 10 (step 4)")]:
        off, ft = [], []
        for _, exp, k_off, k_ft in rows:
            v = json.load(open(EXP / exp / "validation.json"))["validation"]
            off.append(v[k_off]["recall@1"])
            ft.append(v[k_ft]["recall@1"])
        x = np.arange(len(rows))
        for dx, vals, color, label in [(-0.2, off, LIGHT, "off the shelf"), (0.2, ft, BLUE, "fine-tuned")]:
            bars = ax.bar(x + dx, vals, 0.38, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x, [r[0] for r in rows], fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0.65, 0.83)
    axes[0].set_ylabel("Recall@1, validation")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Fine-tuning improves every model we tried, at both stages", x=0.02, ha="left", color=INK, fontsize=12)
    save(fig, "fig4_finetune_each_model.png")


def fig_question_writers():
    s = pd.read_csv(ROOT / "results/test/test_scores.csv")
    s = s.groupby(["source", "condition"])["recall@1"].mean().unstack(0)
    conds = [c for c, _ in STEPS]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(conds))
    w = 0.38
    for i, (src, label, color) in enumerate([("llama", "Llama questions (test writer)", BLUE),
                                              ("qwen", "Qwen questions (training writer)", ORANGE)]):
        vals = s.loc[conds, src].to_numpy()
        bars = ax.bar(x + (i - 0.5) * w, vals, w - 0.03, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [s_ for _, s_ in STEPS])
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Recall@1, test")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("The same gains whichever LLM wrote the questions", loc="left", color=INK, fontsize=12)
    save(fig, "fig5_question_writers.png")


def fig_errors():
    cats = [("Same topic, both pages fit the question", 24), ("Answer title says little", 7),
            ("Needs world knowledge", 7), ("Near-duplicate page", 6), ("Question dropped the key name", 3),
            ("Broken question or weak label", 3)]
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    y = np.arange(len(cats))[::-1]
    ax.barh(y, [c[1] for c in cats], color=BLUE, height=0.62)
    for yi, (_, n) in zip(y, cats):
        ax.text(n + 0.3, yi, f"{n} ({n * 2}%)", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [c[0] for c in cats])
    ax.set_xlim(0, 28)
    ax.set_xlabel("Misses (of 50 sampled)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Half the misses are pages the question cannot tell apart", loc="left", color=INK, fontsize=11)
    save(fig, "fig6_error_categories.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("fig*.png"):
        old.unlink()
    ci = condition_ci()
    fig_steps(ci)
    fig_recall_at_k()
    fig_rerank_k()
    fig_finetune_each_model()
    fig_question_writers()
    fig_errors()
