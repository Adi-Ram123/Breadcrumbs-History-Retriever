# Plan

Two parts, in order. Part 2 starts only when Part 1's report is final.

## Part 1: the experiment (this repo)

**Research question:** when people ask an assistant about their browsing history in their own words, which retriever returns the page they mean? Each question ranks its own 200-page history (the answer, 100 BM25 look-alikes of it, 99 random pages). Part 1 explores the approach only: there is no time handling and there are no synthetic timestamps.

**Two stages, each judged by its own job.**
- The **first stage** (bi-encoder) must get the answer into a short list. It is judged by Recall@k for k = 10, 20, 50 and 100, which is the ceiling for the reranker.
- The **reranker** must put the answer first. It is judged by Recall@1, with MRR breaking ties, after reordering the first stage's top k.
- Recall@5 is also reported. The report shows reranker latency for each k, so that Part 2 can choose k for the app.

### Questions

- **Training questions:** Qwen rewrites of the ORCAS searches (unchanged).
- **Validation and test questions:** Llama rewrites, regenerated with a **neutral prompt**. The earlier prompt forbade copying three or more title words and asked for loose descriptions, which handicapped BM25 by design. The two rules are removed; everything else in the prompt stays.
- **Check sets on the same pages:** the Qwen questions (same style as training) and the original ORCAS keyword queries (typed by real people). A model only counts as better if it does not lose on the keyword queries.

### Conditions

| # | Condition |
|---|---|
| 1 | BM25 |
| 2 | all-MiniLM-L6-v2, off the shelf (the app's current default embedder) |
| 3 | bge bi-encoder, off the shelf |
| 4 | bge bi-encoder, fine-tuned (look-alike hard negatives) |
| 5 | Hybrid: BM25 + fine-tuned bi-encoder, reciprocal rank fusion |
| 6 | Fine-tuned bi-encoder + bge reranker **off the shelf** on its top 10, scores fused |
| 7 | Fine-tuned bi-encoder + bge reranker **fine-tuned** on its top 10, scores fused (**final model**) |
| 8 | Claude, hosted, ranking the whole 200-page history (reference only, never part of the app) |

- **Why two models in 6 and 7:** the bi-encoder embeds every page once, so a question costs one dot product per page and scales to large histories. The reranker (a cross-encoder) reads the question and a page together, which is more accurate but costs one model pass per page, so it only reorders the bi-encoder's top 10. Both come from the BGE family.
- **Model sizes, chosen once on validation and fixed before any result is seen:**
  - bi-encoder: `bge-base-en-v1.5` (110M) or `bge-large-en-v1.5` (335M), chosen by Recall@k. Large is used only if it wins by at least 2 points without losing on the keyword queries.
  - reranker: `bge-reranker-base` (278M) or `bge-reranker-v2-m3` (568M), chosen by Recall@1, with reranking under about 250 ms per question at k = 10 on the Mac GPU.
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

The app becomes a retrieval-augmented generation (RAG) system. One trained retriever makes every language decision, and the LLM only writes the reply. There is no parser and no combiner. Nothing is pushed until the owner reviews it. Changes to Daniel's areas are kept in separate commits.

```text
question + now ─► first stage: top k by meaning + recent entries ─► reranker reads "title | url | visited 1 day ago · Tue evening · Nov 24"
              ─► top 10 with visit times ─► LLM writes the answer, citing only those entries
```

1. **Integration.**
   - Export the Part 1 bi-encoder and reranker to `ml/models/`.
   - Implement `load_embedder`, with the query prefix on the question side.
   - Add a reranker step in the executor.
   - The answer writer receives the top 10 entries with their visit times.
2. **Time handled by the retriever.**
   - Page vectors are computed once, so the first stage cannot judge time itself. Its candidates are therefore the top k by meaning plus the recent entries, so that time questions are not missed.
   - The reranker reads each candidate's visit time written as text when the question is asked ("1 day ago", weekday, time of day, date). It matches that to phrases such as "yesterday" or "last week".
3. **Real training on the user's data.**
   - The reranker is fine-tuned on the user's history: re-finding events and later side-panel clicks.
   - Time-phrase questions are written for real visit times. They teach what "yesterday" means, not the user's habits.
   - Training uses earlier months and testing uses later ones.
4. **Evaluation.**
   - The app's dev questions, including date questions written by the user, scored by Recall@1 and Recall@5 and by date accuracy.
   - The comparison is against stock all-MiniLM and against a Claude call given the same history.
