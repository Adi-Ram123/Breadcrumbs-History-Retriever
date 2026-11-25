# Where off-the-shelf bge-small misses Recall@1

Llama test questions (2,000) plus validation (500); 821 questions where the answer is not ranked first.

- Not caused by duplicates: the page ranked first is an exact copy of the answer in 0.4% of misses, has the same title in 8.9%, and is on the same site in 24.6%.
- The answer is usually close: rank 2 in 37% of misses, top 5 in 68%.
- Test Recall@k: @1 0.676, @2 0.794, @3 0.848, @5 0.896, @10 0.939, @20 0.969, @50 0.992 (`recall_at_k.png`).
- In 25 sampled misses (`misses_sample.csv`), about half need the question and page read together or world knowledge (Quebec city vs province, AC/DC -> Angus Young, tomb sweeping -> Qingming, a recipe vs the Wikipedia article). About 10 cannot be told apart from the question alone (vikings.com vs sbnation, drugs.com vs Wikipedia), which suggests Recall@1 tops out around 0.85.

Next step: rerank the bi-encoder's top-k with a cross-encoder, which reads question and page together. With k = 20 the ceiling is 0.969.
