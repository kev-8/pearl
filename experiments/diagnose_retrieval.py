"""Inspect archive candidate stages without writing evidence or trace files.

This is a paid, manual diagnostic: each query calls Cohere Embed, Pinecone,
and Cohere Rerank through the same production search implementation.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modeling  # noqa: E402


def inspect(query: str) -> None:
    index = modeling.get_index()
    captured: dict[str, list[dict]] = {}
    original_filter = modeling.filter_low_quality
    original_rerank = modeling.rerank_matches

    class RecordingIndex:
        def query(self, **kwargs):
            response = index.query(**kwargs)
            captured["initial"] = list(response.get("matches", []))
            return response

    def record_filter(matches):
        kept = original_filter(matches)
        captured["after_quality"] = list(kept)
        return kept

    def record_rerank(routed_query, matches):
        ranked = original_rerank(routed_query, matches)
        captured["final"] = list(ranked)
        return ranked

    with (
        patch.object(modeling, "get_index", return_value=RecordingIndex()),
        patch.object(modeling, "filter_low_quality", side_effect=record_filter),
        patch.object(modeling, "rerank_matches", side_effect=record_rerank),
    ):
        modeling.search_archive(query)

    initial = captured["initial"]
    quality_ids = {str(item.get("id")) for item in captured["after_quality"]}
    final_ids = {str(item.get("id")) for item in captured["final"]}
    print(
        f"\nQuery: {query}\n"
        f"Initial: {len(initial)}, after quality: {len(quality_ids)}, "
        f"final: {len(final_ids)}"
    )
    for rank, match in enumerate(initial, start=1):
        vector_id = str(match.get("id", ""))
        metadata = match.get("metadata") or {}
        text = " ".join(str(metadata.get("text") or "").split())[:140]
        stage = (
            "final" if vector_id in final_ids else
            "quality-drop" if vector_id not in quality_ids else
            "rerank-drop"
        )
        print(f"{rank:2d} {stage:11s} {vector_id} {metadata.get('sqldate', '')} {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()
    for query in args.query:
        inspect(query)


if __name__ == "__main__":
    main()
