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
- **Model choice: accuracy first, then speed.** Earlier rounds showed that small models (bge-small, MiniLM) lose Recall@1 for lack of capacity. Each stage therefore fine-tunes its 3 most accurate candidates. Every candidate is widely used in apps and loads without custom code. The off-the-shelf screening is in `results/experiments/screening/screening.md` (an appendix table).
  - **First stage** (judged by Recall@10 and Recall@50):
    - **`mxbai-embed-large-v1`** (335M). Best Recall@10 (0.978) and Recall@50 0.996.
    - **`snowflake-arctic-embed-l-v2.0`** (568M). Best Recall@1 (0.774) and MRR (0.848), with Recall@10 0.972.
    - **`e5-large-v2`** (335M). Best Recall@50 (0.998) and keyword Recall@50 (0.992).
  - **Fit with the app:** all three are encoder bi-encoders with standard pooling. Each exports as a sentence-transformers model with a `retriever.json` and needs no custom code. They stay text only, as Part 2 requires.
  - **Reranker** (judged by Recall@1; fine-tuned on the chosen first stage's top-k mistakes). Part 2's preferred design (Option 2) joins the reranker's pooled (question, page) vector with a feature vector in one small layer. So every candidate is an encoder cross-encoder with a pooled vector, and LLM rerankers are excluded.
    - **`bge-reranker-v2-m3`** (568M, XLM-RoBERTa). Best off the shelf: Recall@1 0.770, 66 ms at k = 10.
    - **`mxbai-rerank-large-v1`** (435M, DeBERTa-v3-large). Mixedbread reports it is among the strongest open rerankers on BEIR. It still needs an off-the-shelf screen first, which takes minutes.
    - **`gte-reranker-modernbert-base`** (149M, ModernBERT). Recall@1 0.754 at 30 ms, within 1.6 points of v2-m3, with an 8k context that fits page-text chunks.
  - **Selection:**
    - The most accurate fine-tuned candidate wins.
      - Reranker: Recall@1, then MRR.
      - First stage: Recall@10 and Recall@50, but when they are saturated (all candidates within 1 point), Recall@1 and MRR on the Llama and keyword questions decide.
    - Speed decides only when it is a real application constraint: the winner breaks the latency budget at the app's k (about 250 ms per question for the reranker), or it is at least 1.5× slower than a candidate within 0.5 points.
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

**Agent token budget.** Training runs unattended under a driver script that retries on failure and writes a final status line. Agents check the run's log at most every 30 minutes, with one short command and a one-line report. They do not watch the run, poll it in tight loops or reread large notebook outputs.
4. Score every condition once on the test set (2,000 questions) with the Llama, Qwen and keyword questions. Run a paired bootstrap against conditions 1, 2 and 3, and measure CPU and GPU time per question.
5. Claude reference: a random 500 test Llama questions, sent through the Batch API with the whole 200-page history. Our final model is scored on the same 500. It needs an Anthropic API key, stored SOPS-encrypted and run with `with-secrets`.
6. Write the report: `results/scores.md` and the README.

## Schedule: everything done by end of Friday 2026-11-27

- **Part 1:**
  - The reranker winner still gets 3 seeds.
  - The step 4 test scoring is built in worktree `test-eval` while the rerankers train, and it runs as soon as the reranker seeds finish.
  - The failed training attempts are diagnosed, so the remaining runs avoid them.
- **Part 2 starts now, in parallel**, and no longer waits for the Part 1 report.
  - Integration (step 1) is built against the known first stage (mxbai). The reranker winner is plugged in when it is picked.
  - Option 1 (features as text) is trained first. Option 2 (features as data) trains overnight. If Option 2 is not ready by Friday midday, Option 1 ships and the report says Option 2 is pending.
  - Real training uses re-finding events only. Side-panel clicks are added later, because they need weeks of use.
  - The Claude comparison is skipped for now, in both parts; the owner will decide on it later.
  - **Before vs after.** Part 2's headline comparison is the trained app against the app as it was before this work: its old pipeline, unchanged, on the same test questions and synthetic history.
  - **No user input.** Everything runs autonomously, and no step waits for the owner. The dev and test questions are generated, not hand-written.
  - **Question mix (training and evaluation):** generated from the user's real history, with labelled types, so results are reported for each type, not only overall.
    - Topic only, no constraint: "the ramen recipe I looked at".
    - Time only: "what did I open yesterday evening?".
    - Topic plus time: "the flight deal from last Tuesday".
    - Kind: search vs visit ("what did I search about tax forms?").
    - Site or source: "the GitHub issue about the login bug".
    - Order and recency: "the last movie I watched", "the first page I opened this morning".
    - Frequency: "the docs page I keep going back to".
    - Vague or paraphrased, with no title words.
    - No answer: the page is not in the history, and the system should say so rather than cite something.
  - **Writers:** training questions are written by the local Qwen. Dev and test questions are written by a different local model (Llama, as in Part 1), so the reranker cannot learn one writer's habits. Train on earlier months and test on later ones.
  - **Report per type:** Recall@1, Recall@5 and first-stage recall at k for each question type, date accuracy for the types with a time constraint, and the no-answer rate for the no-answer type. All are compared against stock all-MiniLM.
- **GPU:** every GPU job still goes through `$TMPDIR/gpu_run.sh`. The Part 1 runs go first.

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
