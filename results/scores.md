# Scores so far

Each question ranks its own 200-page history (1 answer + 100 BM25 look-alikes of the answer page + 99 random pages).
Primary metric is Recall@1: off-the-shelf bge already passes 85% validation Recall@5, so a 10-point Recall@5 gain is impossible.
Full numbers, including nDCG@10, are in `scores.csv`. The `rrf_bm25_bge_base` hybrid is computed from the two rankings files with 1 / (60 + rank).

## Validation, Llama questions (500)

| Model | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.476 | 0.708 | 0.582 |
| bge-small, off the shelf | 0.654 | 0.896 | 0.764 |
| BM25 + bge-small, RRF hybrid | 0.550 | 0.816 | 0.675 |
| bge-small fine-tuned, all rows, hard negatives (seed 0) | 0.692 | 0.892 | 0.785 |
| bge-small fine-tuned, lowest 10% train_score dropped, hard negatives (seed 0) | 0.694 | 0.898 | 0.786 |

## Test (2,000 per source)

| Model | Llama R@1 | Llama R@5 | Qwen R@1 | Qwen R@5 | Keyword R@1 | Keyword R@5 |
|---|---|---|---|---|---|---|
| BM25 | 0.486 | 0.717 | 0.478 | 0.700 | 0.410 | 0.637 |
| bge-small, off the shelf | 0.676 | 0.896 | 0.644 | 0.874 | 0.591 | 0.842 |
| BM25 + bge-small, RRF hybrid | 0.580 | 0.818 | 0.568 | 0.800 | 0.462 | 0.742 |

Fine-tuned models have no test scores yet: training was stopped after two selection runs to revise the approach.

## Experiments (validation, Llama questions; 50k training rows, bf16, seed 0)

Each experiment changes one thing from `baseline-50k`. Full settings and numbers are in `results/experiments/<name>/validation.json`.

| Experiment | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| baseline-50k (question BM25 negative, lr 2e-5, 1 epoch, batch 128) | 0.690 | 0.902 | 0.783 |
| url-words (URL split into words) | 0.682 | 0.894 | 0.780 |
| lookalike-negatives (1 look-alike of the positive page) | **0.708** | **0.920** | 0.798 |
| lookalike-negatives-3 (3 look-alikes) | **0.708** | 0.918 | **0.800** |
| train-harder-ep, lr 5e-5, epoch 1 / 2 / 3 | 0.676 / 0.670 / 0.668 | 0.898 / 0.888 / 0.896 | 0.773 / 0.771 / 0.767 |
| train-harder-batch, lr 1e-4, batch 1024 cached loss, epoch 1 / 2 / 3 | 0.692 / 0.690 / 0.694 | 0.906 / 0.908 / 0.904 | 0.787 / 0.788 / 0.789 |

With url-words, BM25 rose to 0.486 / 0.714 / 0.595 and off-the-shelf bge-small to 0.668 / 0.892 / 0.771.

## Reranker (Part 1, step 3; validation, 500 questions per set, Recall@1)

Fine-tuned as sentence-transformers CrossEncoders on 50k rows from mxbai-embed-large-v1's top k (20k for bge-reranker-v2-m3, as planned). mxbai-rerank-large-v1 was screened off the shelf (0.710 at k = 10 on e5's top k) and dropped from training by the coordinator. App k = 10.

Selection on mxbai's top k (first stage alone 0.760, keyword 0.656):

| Reranker | k=10 | k=20 | k=50 | ms/q (k=10) | Qwen | Keyword | Off the shelf |
|---|---|---|---|---|---|---|---|
| gte-reranker-modernbert-base (seed 0) | **0.804** | 0.802 | 0.800 | 40 | 0.806 | 0.662 | 0.754 |
| bge-reranker-v2-m3 | 0.782 | | | 79 | | 0.658 | 0.770 |

gte won by more than 1 point and is faster. Its seeds 1 and 2 scored 0.802 and 0.812. v2-m3 trained with the cached MNRL loss (mini-batch 16, same gradient), eager attention and batch 16, peaking at 24 GB; at batch 32 it used 74 GB.

On snowflake's top k (the chosen first stage: alone 0.790, keyword 0.682). Rerankers were trained on mxbai's candidates. "Fused" is `w_bi * z(first stage) + z(reranker)` within the top 10, with `w_bi` chosen on 1,000 held-out training rows (pool rows 50k–51k, questions plus keyword queries), never on validation:

| Reranker | Alone | Alone keyword | w_bi | Fused | Fused Qwen | Fused keyword |
|---|---|---|---|---|---|---|
| gte seed 0 | 0.804 | 0.658 | 1.25 | 0.816 | 0.824 | 0.694 |
| gte seed 1 | 0.802 | 0.670 | 0.6 | 0.814 | 0.834 | 0.710 |
| gte seed 2 | 0.810 | 0.680 | 1.0 | 0.818 | 0.826 | 0.716 |
| bge-reranker-v2-m3 | 0.780 | 0.652 | 1.0 | 0.808 | 0.824 | 0.686 |

Without fusion, gte loses to snowflake on keyword queries (0.658 vs 0.682), which breaks PLAN.md's keyword rule. With fusion, all three seeds pass (0.694–0.716), so nothing was retrained. Winner: gte seed 0, fused at k = 10 with w_bi 1.25. The k = 10 cost is about 31 ms per question.
