# reranker-keywords: stopped before its first validation point

Settings: MiniLM-L12 cross-encoder, RERANK_TOP_K = 10, RERANK_KEYWORDS = True, 50k question rows + 15,039 keyword-query
examples (65,039 examples, about 4,065 steps at batch 16), look-alike bi-encoder first stage, bf16.

Stopped by the coordinator (experiment dropped from the plan) at step ~1,250 of ~4,065, before the 1/3 validation at
step 1,355, so there are no reranker validation numbers. First stage on validation (unchanged, identical to
cross-encoder-minilm12): Recall@10 0.952 (the most reranking its top 10 can reach); Recall@1 0.708 / 0.708 / 0.626
(Llama / Qwen / keyword).
