# Off-the-shelf screening (500 validation questions, mps)

## First stage

Llama questions; keyword Recall@50 in the last column. BM25 times are on the CPU, the others on the GPU.

| Model | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR | ms/entry | ms/question | Keyword R@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.522 | 0.766 | 0.832 | 0.884 | 0.938 | 0.976 | 0.632 | 0.011 | 0.2 | 0.870 |
| all-MiniLM-L6-v2 | 0.672 | 0.896 | 0.952 | 0.974 | 0.994 | 0.996 | 0.776 | 0.325 | 7.6 | 0.984 |
| bge-base-en-v1.5 | 0.734 | 0.934 | 0.970 | 0.982 | 0.994 | 1.000 | 0.826 | 1.432 | 11.5 | 0.988 |
| bge-large-en-v1.5 | 0.742 | 0.942 | 0.968 | 0.980 | 0.996 | 0.998 | 0.832 | 4.439 | 24.5 | 0.992 |
| mxbai-embed-large-v1 | 0.752 | 0.958 | 0.978 | 0.986 | 0.996 | 0.998 | 0.838 | 1.443 | 21.3 | 0.990 |
| e5-large-v2 | 0.730 | 0.944 | 0.966 | 0.986 | 0.998 | 0.998 | 0.822 | 4.605 | 23.0 | 0.992 |
| snowflake-arctic-embed-l-v2.0 | 0.774 | 0.942 | 0.972 | 0.982 | 0.992 | 0.996 | 0.848 | 4.459 | 13.5 | 0.990 |
| Qwen3-Embedding-0.6B | 0.712 | 0.908 | 0.946 | 0.974 | 0.992 | 1.000 | 0.803 | 3.008 | 16.1 | 0.984 |

## Rerankers on the top k of e5-large-v2

Llama questions, reranker score alone; keyword Recall@1 in the last column. GPU time per question, k pairs in one batch.

| Reranker | k | R@1 | R@5 | MRR | ms/question | Keyword R@1 |
|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L6-v2 | 10 | 0.722 | 0.930 | 0.810 | 6 | 0.584 |
| ms-marco-MiniLM-L6-v2 | 20 | 0.724 | 0.916 | 0.809 | 8 | 0.582 |
| ms-marco-MiniLM-L6-v2 | 50 | 0.724 | 0.912 | 0.806 | 17 | 0.580 |
| bge-reranker-base | 10 | 0.684 | 0.908 | 0.791 | 22 | 0.580 |
| bge-reranker-base | 20 | 0.670 | 0.904 | 0.777 | 41 | 0.570 |
| bge-reranker-base | 50 | 0.662 | 0.892 | 0.766 | 106 | 0.554 |
| gte-reranker-modernbert-base | 10 | 0.754 | 0.936 | 0.832 | 30 | 0.632 |
| gte-reranker-modernbert-base | 20 | 0.748 | 0.932 | 0.828 | 55 | 0.632 |
| gte-reranker-modernbert-base | 50 | 0.746 | 0.918 | 0.822 | 145 | 0.620 |
| bge-reranker-v2-m3 | 10 | 0.770 | 0.942 | 0.846 | 66 | 0.622 |
| bge-reranker-v2-m3 | 20 | 0.762 | 0.938 | 0.841 | 132 | 0.618 |
| bge-reranker-v2-m3 | 50 | 0.760 | 0.936 | 0.838 | 348 | 0.618 |
| Qwen3-Reranker-0.6B | 10 | 0.748 | 0.944 | 0.837 | 162 | 0.614 |
| Qwen3-Reranker-0.6B | 20 | 0.746 | 0.942 | 0.836 | 304 | 0.612 |
| Qwen3-Reranker-0.6B | 50 | 0.746 | 0.938 | 0.835 | 757 | 0.608 |
| mxbai-rerank-large-v1 (CPU time) | 10 | 0.710 | 0.938 | 0.810 | 608 | 0.578 |
| mxbai-rerank-large-v1 (CPU time) | 20 | 0.692 | 0.936 | 0.799 | 1138 | 0.574 |
| mxbai-rerank-large-v1 (CPU time) | 50 | 0.682 | 0.920 | 0.790 | 2779 | 0.570 |

mxbai-rerank-large-v1 was screened later on the CPU (the GPU was busy), loaded in float32, on the same 500 validation
questions and the top k of e5-large-v2 recomputed on the CPU (Llama Recall@1 0.728, Recall@10 0.966, Recall@50 0.998,
against 0.730, 0.966 and 0.998 on the GPU). Its ms/question is CPU time and is not comparable with the GPU rows.
