"""Figures and per-condition confidence intervals for results/report.md.

Reads only saved results (no model runs):
  results/test/answer_ranks.parquet, results/test/test_scores.csv, results/test/rerank_k.csv, results/test/speed.csv,
  results/experiments/*/validation.json, results/experiments/screening/screening.md (numbers copied below).
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

# Palette: validated categorical order (blue, orange, aqua, yellow); ink and grid colors.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})

NAMES = {
    1: "1  BM25",
    2: "2  all-MiniLM-L6-v2, off the shelf",
    3: "3  snowflake-arctic-embed-l-v2.0, off the shelf",
    4: "4  snowflake, fine-tuned",
    5: "5  BM25 + 4, RRF",
    6: "6  4 + gte reranker, off the shelf",
    7: "7  4 + gte reranker, fine-tuned (final)",
    9: "A  mxbai-embed-large-v1, fine-tuned",
}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def condition_ci(n_boot=10_000, seed=0):
    """Recall@1 per condition and source, averaged over seeds per question, with a 95% bootstrap CI."""
    r = pd.read_parquet(ROOT / "results/test/answer_ranks.parquet")
    r["hit"] = (r["rank"] == 1).astype(float)
    rng = np.random.default_rng(seed)
    rows = []
    for (src, cond), g in r.groupby(["source", "condition"]):
        per_q = g.groupby("qid")["hit"].mean().to_numpy()
        idx = rng.integers(0, len(per_q), size=(n_boot, len(per_q)))
        boot = per_q[idx].mean(axis=1)
        rows.append({"source": src, "condition": cond, "n": len(per_q), "seeds": g["seed"].nunique(),
                     "recall@1": per_q.mean(), "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "condition_ci.csv", index=False)
    return df


def fig_conditions(ci):
    d = ci[ci.source == "llama"].copy()
    d["order"] = d.condition.map({1: 1, 2: 2, 3: 3, 9: 4, 4: 5, 5: 6, 6: 7, 7: 8})
    d = d.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = [BLUE if c == 7 else (GRAY if c in (1, 5) else "#86b6ef") for c in d.condition]
    y = np.arange(len(d))
    ax.barh(y, d["recall@1"], color=colors, height=0.62)
    ax.errorbar(d["recall@1"], y, xerr=[d["recall@1"] - d.ci_low, d.ci_high - d["recall@1"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for yi, v, hi in zip(y, d["recall@1"], d.ci_high):
        ax.text(hi + 0.008, yi, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [NAMES[c] for c in d.condition])
    ax.set_xlim(0.4, 0.9)
    ax.set_xlabel("Recall@1 on 2,000 Llama-written test questions")
    ax.grid(axis="y", visible=False)
    ax.set_title("Each stage adds accuracy; the BM25 hybrid hurts", loc="left", color=INK, fontsize=12)
    save(fig, "fig1_test_conditions.png")


def fig_sources():
    s = pd.read_csv(ROOT / "results/test/test_scores.csv")
    s = s.groupby(["source", "condition"])["recall@1"].mean().unstack(0)
    conds = [1, 2, 3, 4, 7]
    srcs = [("llama", "Llama questions", BLUE), ("qwen", "Qwen questions", ORANGE), ("keyword", "Keyword queries (people)", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(conds))
    w = 0.26
    for i, (src, label, color) in enumerate(srcs):
        vals = s.loc[conds, src].to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, ["BM25", "MiniLM\noff the shelf", "snowflake\noff the shelf", "snowflake\nfine-tuned", "final\n(+ reranker)"])
    ax.set_ylim(0.3, 0.97)
    ax.set_ylabel("Test Recall@1 (mean of seeds)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.set_title("Gains hold on all three question sets, including real keyword queries", loc="left", color=INK, fontsize=12)
    save(fig, "fig2_three_question_sets.png")


def fig_first_stage_selection():
    models = [("mxbai-embed-large-v1", "first-stage-mxbai", "mxbai-embed-large-v1\n335M"),
              ("snowflake-arctic-embed-l-v2.0", "first-stage-snowflake", "snowflake-arctic-\nembed-l-v2.0, 568M"),
              ("e5-large-v2", "first-stage-e5", "e5-large-v2\n335M")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (suffix, title) in zip(axes, [("", "Llama questions"), ("_keyword", "Keyword queries")]):
        off, ft = [], []
        for base, exp, _ in models:
            v = json.load(open(EXP / exp / "validation.json"))["validation"]
            off.append(v[base + suffix]["recall@1"])
            ft.append(v[f"exp_{exp}{suffix}"]["recall@1"])
        x = np.arange(len(models))
        for dx, vals, color, label in [(-0.2, off, "#86b6ef", "off the shelf"), (0.2, ft, BLUE, "fine-tuned (seed 0)")]:
            bars = ax.bar(x + dx, vals, 0.38, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(x, [m[2] for m in models], fontsize=9)
        ax.set_title(title, color=INK, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0.5, 0.85)
    axes[0].set_ylabel("Validation Recall@1 (500 questions)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("First stage: fine-tuning helps all three; snowflake leads on both sets", x=0.02, ha="left", color=INK, fontsize=12)
    save(fig, "fig3_first_stage_selection.png")


def fig_screening():
    # From results/experiments/screening/screening.md (validation, off the shelf, Llama questions).
    first = [("BM25", 0.522, 0.832), ("all-MiniLM-L6-v2 (22M)", 0.672, 0.952), ("Qwen3-Embedding-0.6B", 0.712, 0.946),
             ("e5-large-v2 (335M)", 0.730, 0.966), ("bge-base-en-v1.5 (110M)", 0.734, 0.970),
             ("bge-large-en-v1.5 (335M)", 0.742, 0.968), ("mxbai-embed-large-v1 (335M)", 0.752, 0.978),
             ("snowflake-arctic-embed-l-v2.0 (568M)", 0.774, 0.972)]
    rer = [("ms-marco-MiniLM-L6-v2", 0.722, 6), ("bge-reranker-base", 0.684, 22), ("gte-reranker-modernbert-base", 0.754, 30),
           ("bge-reranker-v2-m3", 0.770, 66), ("Qwen3-Reranker-0.6B", 0.748, 162)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(first))
    chosen = {"snowflake-arctic-embed-l-v2.0 (568M)", "mxbai-embed-large-v1 (335M)", "e5-large-v2 (335M)"}
    ax.barh(y, [f[1] for f in first], color=[BLUE if f[0] in chosen else GRAY for f in first], height=0.62)
    for yi, f in zip(y, first):
        ax.text(f[1] + 0.006, yi, f"{f[1]:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [f[0] for f in first], fontsize=9)
    ax.set_xlim(0.45, 0.82)
    ax.set_xlabel("Validation Recall@1, off the shelf")
    ax.grid(axis="y", visible=False)
    ax.set_title("First-stage screen (blue = fine-tuned next)", loc="left", color=INK, fontsize=11)
    ax = axes[1]
    for name, r1, ms in rer:
        c = BLUE if name in ("gte-reranker-modernbert-base", "bge-reranker-v2-m3") else GRAY
        ax.scatter(ms, r1, s=70, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(name, (ms, r1), textcoords="offset points", xytext=(6, 5), fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("GPU ms per question at k = 10 (log scale)")
    ax.set_ylabel("Validation Recall@1, reranker alone")
    ax.set_xlim(4, 600)
    ax.set_ylim(0.67, 0.79)
    ax.set_title("Reranker screen on e5's top 10", loc="left", color=INK, fontsize=11)
    save(fig, "fig4_screening.png")


def fig_reranker_selection():
    rows = []
    for name, exp in [("gte-reranker-modernbert-base (149M)", "ft-gte-reranker-modernbert-base-on-snowflake"),
                      ("bge-reranker-v2-m3 (568M)", "ft-bge-reranker-v2-m3-on-snowflake")]:
        v = json.load(open(EXP / exp / "validation.json"))["validation"]
        rows.append((name, v["first_stage"]["recall@1"], v["pretrained_k10"]["recall@1"], v["reranker_k10"]["recall@1"]))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(rows))
    for dx, i, color, label in [(-0.27, 1, GRAY, "first stage alone (snowflake fine-tuned)"),
                                (0, 2, "#86b6ef", "+ reranker off the shelf"), (0.27, 3, BLUE, "+ reranker fine-tuned")]:
        vals = [r[i] for r in rows]
        bars = ax.bar(x + dx, vals, 0.25, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylim(0.7, 0.84)
    ax.set_ylabel("Validation Recall@1, top 10\n(reranker score alone)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Fine-tuned gte beats the larger v2-m3 at half the latency (31 vs 73 ms)", loc="left", color=INK, fontsize=12)
    save(fig, "fig5_reranker_selection.png")


def fig_rerank_k():
    k = pd.read_csv(ROOT / "results/test/rerank_k.csv")
    k = k[(k.condition == 7) & (k.kind == "fused")].groupby("k")["recall@1"].mean()
    sp = pd.read_csv(ROOT / "results/test/speed.csv")
    sp = sp[(sp.condition == 7) & sp.k.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(k.index, k.values, color=BLUE, lw=2, marker="o", ms=6)
    ax.axhline(0.806, color=GRAY, lw=1.5, ls="--")
    ax.text(50, 0.8065, "first stage alone", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(0.795, 0.84)
    ax.set_xlabel("k (first-stage candidates reranked)")
    ax.set_ylabel("Test Recall@1, final model")
    ax.set_title("Accuracy is flat past k = 10", loc="left", color=INK, fontsize=11)
    ax = axes[1]
    for dev, color, label in [("mps", BLUE, "Apple GPU (MPS)"), ("cpu", ORANGE, "CPU, 6 threads")]:
        d = sp[sp.device == dev].sort_values("k")
        ax.plot(d.k, d.question_ms, color=color, lw=2, marker="o", ms=6, label=label)
    ax.axhline(250, color=GRAY, lw=1.5, ls="--")
    ax.text(50, 262, "250 ms app budget", ha="right", fontsize=9, color=INK2)
    ax.set_xlabel("k")
    ax.set_ylabel("ms per question")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.set_title("…while latency grows linearly", loc="left", color=INK, fontsize=11)
    save(fig, "fig6_rerank_k.png")


def fig_recall_at_k():
    s = pd.read_csv(ROOT / "results/test/test_scores.csv")
    s = s[s.source == "llama"].groupby("condition").mean(numeric_only=True)
    ks = [1, 5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for c, color, label in [(1, GRAY, "BM25"), (2, ORANGE, "MiniLM, off the shelf"),
                            (3, "#86b6ef", "snowflake, off the shelf"), (4, BLUE, "snowflake, fine-tuned")]:
        vals = [s.loc[c, f"recall@{k}"] for k in ks]
        ax.plot(ks, vals, color=color, lw=2, marker="o", ms=6, label=label)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Test Recall@k (Llama questions)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("The fine-tuned first stage has the answer in its top 10 for 98% of questions", loc="left", color=INK, fontsize=11)
    save(fig, "fig7_first_stage_recall_at_k.png")


def fig_recipe():
    exps = [("BM25", "baseline-50k", "bm25"), ("bge-small, off the shelf", "baseline-50k", "bge_base"),
            ("+ fine-tune (question BM25 negative)", "baseline-50k", "exp_baseline-50k"),
            ("  URL split into words", "url-words", "exp_url-words"),
            ("  lr 5e-5, 3 epochs", "train-harder-ep", "exp_train-harder-ep"),
            ("  lr 1e-4, batch 1024, 3 epochs", "train-harder-batch", "exp_train-harder-batch"),
            ("  look-alike negative of the answer", "lookalike-negatives", "exp_lookalike-negatives"),
            ("  3 look-alike negatives", "lookalike-negatives-3", "exp_lookalike-negatives-3")]
    vals = [json.load(open(EXP / e / "validation.json"))["validation"][k]["recall@1"] for _, e, k in exps]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    y = np.arange(len(exps))[::-1]
    colors = [GRAY, "#86b6ef", BLUE, BLUE, BLUE, BLUE, AQUA, AQUA]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [e[0] for e in exps], fontsize=9.5)
    ax.set_xlim(0.45, 0.75)
    ax.set_xlabel("Validation Recall@1 (bge-small, 50k rows, first question prompt)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Recipe search on a small model: look-alike negatives were the only clear win", loc="left", color=INK, fontsize=11)
    save(fig, "fig8_recipe_experiments.png")


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
    save(fig, "fig9_error_categories.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ci = condition_ci()
    print(ci.round(3).to_string(index=False))
    fig_conditions(ci)
    fig_sources()
    fig_first_stage_selection()
    fig_screening()
    fig_reranker_selection()
    fig_rerank_k()
    fig_recall_at_k()
    fig_recipe()
    fig_errors()
