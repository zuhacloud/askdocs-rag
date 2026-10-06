"""Compare retrieval modes on a QA set (Hit@k, MRR@k, latency).

For each question we know which chunk holds the answer (the "gold" chunk, from generate_qa.py).
For each retrieval mode we check whether that chunk shows up in the top k results, and how high:

    Hit@k  = share of questions whose gold chunk appears anywhere in the top k   (did we find it?)
    MRR@k  = average of 1 / (rank of the gold chunk), 0 if missing              (how near the top?)
             rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2

Hit@k matters because the LLM only sees the top k chunks: a miss means the answer can't be right.
MRR matters because LLMs pay most attention to the first sources.

Stop the API first: embedded Qdrant allows only one process at a time.
Usage: python -m eval.eval_retrieval --qa eval/qa.jsonl --k 5
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from askdocs.pipeline import RAGPipeline
from askdocs.retriever import MODES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="eval/qa.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()

    # One JSON object per line: {"question": ..., "chunk_id": <gold chunk>, "source": ..., "page": ...}
    rows = [json.loads(line) for line in Path(args.qa).read_text(encoding="utf-8").splitlines() if line.strip()]
    pipeline = RAGPipeline()
    # Skip the rerank mode if the reranker is disabled in settings.
    modes = [m for m in MODES if m != "hybrid_rerank" or pipeline.retriever.reranker is not None]

    # Build a Markdown table that can be pasted straight into the README's Results section.
    lines = [
        f"Retrieval on {len(rows)} questions, k={args.k}\n",
        f"| Mode | Hit@{args.k} | MRR@{args.k} | Latency / query |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        # One untimed warm-up query: the first GPU call is slow (CUDA initialisation),
        # which would unfairly inflate the latency of whichever mode runs first.
        pipeline.retrieve(rows[0]["question"], mode, args.k)
        hits, reciprocal_ranks = 0, 0.0
        start = perf_counter()
        for row in rows:
            ids = [c.id for c in pipeline.retrieve(row["question"], mode, args.k)]
            if row["chunk_id"] in ids:
                hits += 1
                # .index() is 0-based, so +1 gives the human rank.
                reciprocal_ranks += 1 / (ids.index(row["chunk_id"]) + 1)
        latency_ms = (perf_counter() - start) / len(rows) * 1000
        lines.append(f"| {mode} | {hits / len(rows):.3f} | {reciprocal_ranks / len(rows):.3f} | {latency_ms:.0f} ms |")

    report = "\n".join(lines)
    print(report)
    Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
