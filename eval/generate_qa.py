"""Build a synthetic QA set: sample indexed chunks and have the LLM write one question per chunk.

The source chunk becomes the gold answer for retrieval evaluation. This avoids hand-labelling:
if the LLM wrote a question FROM chunk X, a good retriever should return chunk X for it.

Caveat: synthetic questions tend to reuse the chunk's own words, which flatters keyword (BM25)
search. The prompt asks for paraphrasing to reduce that bias; skim the output and delete bad rows.

Usage: python -m eval.generate_qa --n 50
"""

import argparse
import json
import random
from pathlib import Path

from askdocs.config import get_settings
from askdocs.pipeline import RAGPipeline

PROMPT = """Write one question a user might ask whose answer is contained in the passage below.
- It must be answerable from this passage alone.
- Paraphrase like a real user would; don't copy long phrases from the passage.
- Don't mention "the passage", "the text" or "the document".
Return only the question, nothing else.

Passage:
{text}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--out", default="eval/qa.jsonl")
    parser.add_argument("--min-chars", type=int, default=300, help="skip chunks shorter than this")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Writing questions never uses the reranker, so don't spend time and GPU memory loading it.
    pipeline = RAGPipeline(get_settings().model_copy(update={"use_reranker": False}))
    # Very short chunks (titles, fragments) don't contain enough to ask a meaningful question about.
    chunks = [(cid, p) for cid, p in pipeline.store.iter_all() if len(p["text"]) >= args.min_chars]
    if not chunks:
        raise SystemExit("No chunks indexed. Ingest some documents first.")
    # A fixed seed makes the sample reproducible: the same documents give the same question set.
    random.seed(args.seed)
    sample = random.sample(chunks, min(args.n, len(chunks)))

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for i, (chunk_id, payload) in enumerate(sample, 1):
            question = pipeline.llm.chat([{"role": "user", "content": PROMPT.format(text=payload["text"])}])
            question = question.strip().strip('"')  # models sometimes wrap the question in quotes
            row = {"question": question, "chunk_id": chunk_id, "source": payload["source"], "page": payload["page"]}
            # JSON Lines format: one JSON object per line, easy to append to and stream.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(sample)}] {question}")
    print(f"\nWrote {len(sample)} questions to {out}. Skim them and delete bad ones before evaluating.")


if __name__ == "__main__":
    main()
