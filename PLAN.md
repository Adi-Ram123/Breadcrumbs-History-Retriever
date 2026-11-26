# Plan

Two parts, in order. Part 2 starts only when Part 1's report is final.

## Part 1: the experiment (this repo)

**Research question:** when people ask an assistant about their browsing history in their own words, which retriever returns the page they mean? Each question ranks its own 200-page history (the answer, 100 BM25 look-alikes of it, 99 random pages); Recall@1 is the primary metric, MRR breaks ties, Recall@5 is reported.

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
- **Model sizes, chosen once on validation:** bi-encoder `bge-base-en-v1.5` (110M) or `bge-large-en-v1.5` (335M); reranker `bge-reranker-base` (278M) or `bge-reranker-v2-m3` (568M). The rule is the best validation Recall@1, with reranking under about 250 ms per question at k = 10 on the Mac GPU. If a larger model wins, it is used.
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

Nothing is pushed until the owner reviews it. Changes to Daniel's areas (executor, ports, ingest) are kept in separate commits so they can be agreed with him.

1. **Integration.**
   - Export the final bi-encoder and reranker to `ml/models/`.
   - Implement `load_embedder`, with a query-side encoder (`embed_query`) for the query prefix.
   - Add a `Reranker` port.
   - Executor: SQL time and kind filter → bi-encoder on the full question → deduplicate by page → rerank the top 10 → fuse.
   - The LLM only writes the answer sentence, citing the returned entries. Ranking never depends on it.
2. **Retraining for the app's fields.** The text models stay as trained. A small learned combiner adds the behavioural fields from the `entries` table:
   - days since the last visit;
   - visit count;
   - whether a search came just before the visit;
   - same site as the last visit;
   - later, clicks in the side panel.

   **Labels** come from re-finding events: a search, then a visit to a page seen at least a day earlier. Later they come from side-panel clicks. **Training** fits earlier months and tests on later ones. Dates in a question are handled exactly by the parser's time range and the SQL filter.
3. **Backend.** The retrained stack becomes the app's retriever.
   - **Comparison:** on the app's dev questions, against stock all-MiniLM and against a Claude call over the same candidates.
   - **Metrics:** Recall@1 and Recall@5.
   - **Last check:** your own written questions, once at the end.
