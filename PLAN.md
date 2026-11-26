# Plan

Two parts, in order. Part 2 starts only when Part 1's report is final.

## Part 1: the experiment (this repo)

**Research question:** when people ask an assistant about their browsing history in their own words, which retriever returns the page they mean? Each question ranks its own 200-page history (the answer, 100 BM25 look-alikes of it, 99 random pages). Part 1 explores the approach only: there is no time handling and there are no synthetic timestamps.

**Two stages, each judged by its own job.**
- The **first stage** (bi-encoder) must get the answer into a short list. It is judged by Recall@k for k = 10, 20, 50 and 100, which is the ceiling for the reranker. In the app it must lose almost no correct pages.
- The **reranker** must put the answer first. It is judged by Recall@1, with MRR breaking ties, after reordering the first stage's top k.
- Recall@5 is also reported. The report shows reranker Recall@1 and latency for each k up to 50, so that Part 2 can choose k for the app.

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
| 2 | all-MiniLM-L6-v2, off the shelf (the app's current default embedder) |
| 3 | First-stage bi-encoder, off the shelf |
| 4 | First-stage bi-encoder, fine-tuned (look-alike hard negatives) |
| 5 | Hybrid: BM25 + fine-tuned bi-encoder, reciprocal rank fusion |
| 6 | Fine-tuned bi-encoder + reranker **off the shelf** on its top k, scores fused |
| 7 | Fine-tuned bi-encoder + reranker **fine-tuned** on its top k, scores fused (**final model**) |
| 8 | Claude, hosted, ranking the whole 200-page history (reference only, never part of the app) |

- **Why two models in 6 and 7:** the bi-encoder embeds every page once, so a question costs one dot product per page and scales to large histories. The reranker (a cross-encoder) reads the question and a page together, which is more accurate but costs one model pass per page, so it only reorders the bi-encoder's top 10. The models are picked by the screening below.
- **Model choice: screen widely, fine-tune few, decide once on validation.** Each candidate is first scored off the shelf on the new validation questions (minutes each). Only models that load without running custom code from the model repository are used.
  - **First stage candidates:** `bge-base-en-v1.5` (110M), `bge-large-en-v1.5` (335M), `mxbai-embed-large-v1` (335M), `e5-large-v2` (335M), `snowflake-arctic-embed-l-v2.0` (568M) and `Qwen3-Embedding-0.6B` (600M, built on an LLM).
    - The top 2 by off-the-shelf Recall@10 and Recall@50 are fine-tuned.
    - A larger model is kept only if it wins by at least 2 points without losing on the keyword queries.
  - **Reranker candidates:** `bge-reranker-base` (278M), `bge-reranker-v2-m3` (568M), `Qwen3-Reranker-0.6B` (600M, LLM-based), `gte-reranker-modernbert-base` (149M) and `ms-marco-MiniLM-L6` (22M, speed reference).
    - The top 2 by off-the-shelf Recall@1, among those under about 250 ms per question at k = 10, are fine-tuned on the chosen first stage's top-k mistakes.
  - The screening goes into an appendix table. The results table shows one first stage and one reranker.
- **Fusion:** the bi-encoder and reranker scores are standardised within the top 10 and added. Their weights are picked on held-out training rows, never on validation.
- **Not in the table:** larger reranker experiments that did not improve results (MiniLM-L12, Qwen3-Reranker-0.6B and 4B). One sentence in the report covers them.

### Procedure

1. Regenerate the 2,500 validation and test Llama questions with the neutral prompt.
2. Pick the model sizes on validation.
3. Train conditions 4 and 7 with 3 seeds.
4. Score every condition once on the test set (2,000 questions) with the Llama, Qwen and keyword questions. Run a paired bootstrap against conditions 1, 2 and 3, and measure CPU and GPU time per question.
5. Claude reference: a random 500 test Llama questions, sent through the Batch API with the whole 200-page history. Our final model is scored on the same 500. It needs an Anthropic API key, stored SOPS-encrypted and run with `with-secrets`.
6. Write the report: `results/scores.md` and the README.

## Part 2: the app (`Adi-Ram123/Breadcrumbs-History-Assistant`, local branch `model`)

The app becomes a retrieval-augmented generation (RAG) system. One trained retriever makes every language decision, and the LLM only writes the reply. There is no parser and no combiner. **All app work stays on the local branch `model`. Nothing is ever pushed to GitHub and no PR is opened; the owner decides what to do with the branch.** Changes to Daniel's areas are kept in separate commits.

```text
question + now ─► first stage: fine-tuned bi-encoder, text only, top k by meaning (k chosen for very high recall, e.g. 50)
                  + the most recent entries (so questions about time alone are never missed)
               ─► reranker scores each candidate using the question, the page text AND its history features
               ─► top 10 ─► LLM writes the answer, citing only those entries
```

1. **Integration.**
   - Export the Part 1 bi-encoder and reranker to `ml/models/`.
   - Implement `load_embedder`, with the query prefix on the question side.
   - Add a reranker step in the executor.
   - The answer writer receives the top 10 entries with their visit times.
2. **History features in the reranker. Both ways are tested and the better one is kept.**
   - The first stage stays text only. Its job is high recall: k is set from Part 1's Recall@k curve, so the right page is almost never cut.
   - **Features**, computed when the question is asked: hours since the visit, weekday and hour, visit count, clicks, whether the page was opened from a search, and time on page.
   - **Option 2, features as data (preferred).** The reranker's text vector for (question, page) and a feature vector go into one small layer that outputs the score. It is trained together with the reranker as one model. A new feature is one more number. Because the layer also sees the question's text vector, it can learn that "yesterday" goes with about 24 hours since the visit.
   - **Option 1, features as text (comparison).** The same features are written into the page text ("searched 1 day ago, Tue 9:14 pm | 3 visits | opened from search"), with no model change.
   - **The choice** is made on the app's dev questions by Recall@1, date accuracy and latency. If the two tie, Option 2 is kept.
   - **Known limit:** a question about an old date with no topic ("what did I search on March 3?") depends on the first stage finding something by meaning.
3. **Real training on the user's data.**
   - The reranker (with its features) is fine-tuned on re-finding events from the user's history and later side-panel clicks.
   - Time-phrase questions are written for real visit times, so the reranker learns what "yesterday" means.
   - Training uses earlier months and testing uses later ones.
4. **Evaluation.**
   - The app's dev questions, including date questions written by the user, scored by Recall@1 and Recall@5 and by date accuracy.
   - First-stage recall at the chosen k is also reported.
   - The comparison is against stock all-MiniLM and against a Claude call given the same history.
