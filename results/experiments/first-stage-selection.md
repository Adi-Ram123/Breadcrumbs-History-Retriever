# First-stage selection (validation, 500 questions)

Look-alike negatives (1 per question), 50,000 question rows (drop10) plus keyword rows, batch 128, bf16, 1 epoch, max length 64. Off-the-shelf rows use the same settings in the same run.

| Run | Questions | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR | CPU ms/entry | Training min |
|---|---|---|---|---|---|---|---|---|---|---|
| mxbai-embed-large-v1 off the shelf | llama | 0.752 | 0.956 | 0.978 | 0.986 | 0.996 | 0.998 | 0.838 |  |  |
| mxbai-embed-large-v1 fine-tuned, seed 0 | llama | 0.760 | 0.962 | 0.976 | 0.988 | 0.994 | 0.998 | 0.848 | 16.95 | 11 |
| mxbai-embed-large-v1 off the shelf | keyword | 0.592 | 0.876 | 0.940 | 0.962 | 0.990 | 0.998 | 0.719 |  |  |
| mxbai-embed-large-v1 fine-tuned, seed 0 | keyword | 0.656 | 0.912 | 0.946 | 0.978 | 0.988 | 1.000 | 0.770 | 16.95 | 11 |
| snowflake-arctic-embed-l-v2.0 off the shelf | llama | 0.774 | 0.942 | 0.972 | 0.982 | 0.992 | 0.996 | 0.848 |  |  |
| snowflake-arctic-embed-l-v2.0 fine-tuned, seed 0 | llama | 0.790 | 0.964 | 0.982 | 0.984 | 0.994 | 0.998 | 0.869 | 17.14 | 31 |
| snowflake-arctic-embed-l-v2.0 off the shelf | keyword | 0.630 | 0.906 | 0.948 | 0.966 | 0.990 | 1.000 | 0.753 |  |  |
| snowflake-arctic-embed-l-v2.0 fine-tuned, seed 0 | keyword | 0.682 | 0.926 | 0.960 | 0.974 | 0.994 | 1.000 | 0.791 | 17.14 | 31 |
| e5-large-v2 off the shelf | llama | 0.730 | 0.944 | 0.966 | 0.986 | 0.998 | 0.998 | 0.822 |  |  |
| e5-large-v2 fine-tuned, seed 0 | llama | 0.780 | 0.964 | 0.982 | 0.986 | 0.994 | 0.998 | 0.862 | 17.77 | 22 |
| e5-large-v2 off the shelf | keyword | 0.610 | 0.882 | 0.936 | 0.974 | 0.992 | 1.000 | 0.733 |  |  |
| e5-large-v2 fine-tuned, seed 0 | keyword | 0.662 | 0.926 | 0.960 | 0.976 | 0.988 | 0.998 | 0.776 | 17.77 | 22 |
| mxbai-embed-large-v1 fine-tuned, seed 1 | llama | 0.762 | 0.966 | 0.982 | 0.986 | 0.994 | 0.998 | 0.853 | 17.04 | 34 |
| mxbai-embed-large-v1 fine-tuned, seed 1 | keyword | 0.656 | 0.924 | 0.952 | 0.972 | 0.986 | 0.998 | 0.772 | 17.04 | 34 |
| mxbai-embed-large-v1 fine-tuned, seed 2 | llama | 0.766 | 0.966 | 0.980 | 0.988 | 0.994 | 0.998 | 0.853 | 17.12 | 24 |
| mxbai-embed-large-v1 fine-tuned, seed 2 | keyword | 0.652 | 0.914 | 0.948 | 0.978 | 0.986 | 1.000 | 0.769 | 17.12 | 24 |

Training minutes count the last attempt only. CPU ms per entry is from the fine-tuned model.

## Decision (PLAN.md)

Accuracy = mean of Llama Recall@10 and Recall@50. The winner must not lose on the keyword queries (keyword accuracy within 1 point of the best); among candidates within 1 point of the best accuracy, the fastest wins.

- Keyword-eligible: e5, mxbai, snowflake
- Within 1 point of the best: e5, mxbai, snowflake
- **Winner: `mixedbread-ai/mxbai-embed-large-v1`**, extra seeds: 1, 2

## Notes

- Training data: the rewritten Qwen training questions (all 300,000 rows), the new Llama validation questions.
- snowflake trained with the cached loss (mini batch 32, same batch of 128): without it the run used 69 GB and swapped.
- mxbai seed 0 ran out of GPU memory once and resumed from its checkpoint, so its 11 minutes cover only the second half.
- The speed tie-break rests on CPU times that differ by under 1 ms per entry (16.95 to 17.77, about 5%), close to timing noise.
  On accuracy alone, snowflake leads mxbai seed 0 on every Llama and keyword metric except a tie at Recall@50
  (Llama Recall@1 0.790 against 0.760, Recall@10 0.982 against 0.976, keyword Recall@10 0.960 against 0.946).

## Failed attempts (driver retries)

- mxbai attempt 1: Metal out-of-memory (`MPS command buffer execution failed: Insufficient Memory`) during training at batch 128 without the cached loss. Nothing else held the GPU then. Attempt 4 resumed from the last checkpoint and finished. Seeds 1 and 2 ran with the same settings and did not hit it, so the peak is close to the limit rather than always over it.
- mxbai attempts 2 and 3: a bug in the headless runner (it skipped cell 17, so `dense_speed` was undefined at the speed cell). Fixed in the runner; it affected only this worktree.
- snowflake attempt 1: without the cached loss, batch 128 used about 69 GB and swapped. The kernel was stopped by hand. Attempt 2 used `CACHED_LOSS=True` with mini batch 32.
- The GPU lock was not involved: each attempt held `gpu.lock` alone, and the runner shuts its kernel down before the lock is released.
