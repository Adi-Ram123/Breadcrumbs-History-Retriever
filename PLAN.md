# Plan

The experiment plan, as run. The results are in [`report.md`](report.md).

**Research question:** when people ask an assistant about their browsing history in their own words, which retriever returns the page they mean? Each question ranks its own 200-page history (the answer, 100 BM25 look-alikes of it, 99 random pages). The project explores the approach only: there is no time handling and there are no synthetic timestamps.

**Two stages, each judged by its own job.**
- The **first stage** (bi-encoder) must get the answer into a short list. It is judged by Recall@k for k = 10, 20, 50 and 100, which is the ceiling for the reranker. It must lose almost no correct pages.
- The **reranker** must put the answer first. It is judged by Recall@1, with MRR breaking ties, after reordering the first stage's top k.
- Recall@5 is also reported. The report shows reranker Recall@1 and latency for each k up to 50, so that k can be chosen.

### Questions

- **One prompt for training and test.** The earlier prompt forbade copying three or more title words and asked for loose descriptions, which handicapped BM25 by design. Both rules are removed, and the same neutral rules are used for:
  - the Qwen training questions;
  - the Llama validation and test questions.

  Two different LLMs still write the two sides, so a model cannot learn one writer's habits. Decoding is greedy.
- **Few-shot examples, or none:** a pilot of 300 training rows per LLM compares the two. It measures opening-word shares, site-naming, title-copying and duplicates, plus a manual read of 30 questions per variant. Examples are dropped if they make the questions formulaic.
- **Data size:** all 300k rows are rewritten. The earlier learning curve (bge-small: 0.708 at 50k question rows, 0.724 at about 131k, still rising) shows that more data helps. The first stage trains on every usable row (about 200k questions plus keyword queries). The reranker trains on 50k rows (20k for v2-m3), and more only if its validation curve is still rising. The training filter that dropped questions copying title words is removed, because the prompt now allows them.
- **Check sets on the same pages:** the Qwen questions (same style as training) and the original ORCAS keyword queries (typed by real people). A model only counts as better if it does not lose on the keyword queries.

### Conditions

| # | Condition |
|---|---|
| 1 | BM25 |
| 2 | all-MiniLM-L6-v2, off the shelf |
| 3 | First-stage bi-encoder, off the shelf |
| 4 | First-stage bi-encoder, fine-tuned (look-alike hard negatives) |
| 5 | Hybrid: BM25 + fine-tuned bi-encoder, reciprocal rank fusion |
| 6 | Fine-tuned bi-encoder + reranker **off the shelf** on its top k, scores fused |
| 7 | Fine-tuned bi-encoder + reranker **fine-tuned** on its top k, scores fused (**final model**) |

- **Why two models in 6 and 7:** the bi-encoder embeds every page once, so a question costs one dot product per page and scales to large histories. The reranker (a cross-encoder) reads the question and a page together, which is more accurate but costs one model pass per page, so it only reorders the bi-encoder's top 10. The models are picked by the screening below.
- **Model choice: accuracy first, then speed.** Earlier rounds showed that small models (bge-small, MiniLM) lose Recall@1 for lack of capacity. Each stage therefore fine-tunes its 3 most accurate candidates. Every candidate is widely used in apps and loads without custom code. The off-the-shelf screening is in `results/experiments/screening/screening.md` (an appendix table).
  - **First stage** (judged by Recall@10 and Recall@50):
    - **`mxbai-embed-large-v1`** (335M). Best Recall@10 (0.978) and Recall@50 0.996.
    - **`snowflake-arctic-embed-l-v2.0`** (568M). Best Recall@1 (0.774) and MRR (0.848), with Recall@10 0.972.
    - **`e5-large-v2`** (335M). Best Recall@50 (0.998) and keyword Recall@50 (0.992).
  - **Fit:** all three are encoder bi-encoders with standard pooling. Each loads as a sentence-transformers model and needs no custom code.
  - **Reranker** (judged by Recall@1; fine-tuned on the chosen first stage's top-k mistakes). Every candidate is an encoder cross-encoder; LLM rerankers are excluded.
    - **`bge-reranker-v2-m3`** (568M, XLM-RoBERTa). Best off the shelf: Recall@1 0.770, 66 ms at k = 10.
    - **`mxbai-rerank-large-v1`** (435M, DeBERTa-v3-large). Mixedbread reports it is among the strongest open rerankers on BEIR. It still needs an off-the-shelf screen first, which takes minutes.
    - **`gte-reranker-modernbert-base`** (149M, ModernBERT). Recall@1 0.754 at 30 ms, within 1.6 points of v2-m3, with an 8k context that fits page-text chunks.
  - **Selection:**
    - The most accurate fine-tuned candidate wins.
      - Reranker: Recall@1, then MRR.
      - First stage: Recall@10 and Recall@50, but when they are saturated (all candidates within 1 point), Recall@1 and MRR on the Llama and keyword questions decide.
    - Speed decides only when the winner is at least 1.5× slower than a candidate within 0.5 points.
    - Timing differences under 20% are noise and never decide.
    - **Correction (Nov 26):** the first pick (mxbai, chosen on a 1% speed difference) is replaced by `snowflake-arctic-embed-l-v2.0`, which leads on Recall@1, MRR and the keyword queries.
    - **Data size check:** snowflake is fine-tuned once on all usable question rows (about 200k), not 50k. The 2 extra seeds use whichever size wins.
    - The winner must not lose on the keyword queries.
    - Only the winner of each stage gets the 2 extra seeds.
  - **Screened, then dropped:**
    - `bge-base`, `bge-large`, `bge-reranker-base` and `ms-marco-MiniLM-L6`.
    - `Qwen3-Embedding-0.6B` and `Qwen3-Reranker-0.6B`: they are LLM-based, compete with the local reply LLM for memory, and the reranker has no pooled vector for the feature layer.

    They appear in the appendix table only.
  - The results table shows one first stage and one reranker.
- **Fusion:** the bi-encoder and reranker scores are standardised within the top 10 and added. Their weights are picked on held-out training rows, never on validation.
- **Not in the table:** larger reranker experiments that did not improve results (MiniLM-L12, Qwen3-Reranker-0.6B and 4B). One sentence in the report covers them.

### Procedure

1. Regenerate the 2,500 validation and test Llama questions with the neutral prompt.
2. Fine-tune the 3 first-stage candidates with 1 seed each, and pick one on validation. Screen `mxbai-rerank-large-v1` off the shelf. Then fine-tune the 3 reranker candidates on the chosen first stage's top k with 1 seed each, and pick one.
3. Train 2 more seeds of each winner, so conditions 4 and 7 each have 3 seeds. That makes 10 training runs in total.

4. Score every condition once on the test set (2,000 questions) with the Llama, Qwen and keyword questions. Run a paired bootstrap against conditions 1, 2 and 3, and measure CPU and GPU time per question.
5. Write the report: `results/scores.md` and the README.
