# Assignment 10 - Task 1: semantic article retrieval over the WikiHow indices
# built in Assignment 8 (wikihow.db + title/combined FAISS indices).
from __future__ import annotations

# importing sentence_transformers after torch/transformers crashes on our Windows
# pyarrow build, so we load pyarrow first to initialise its native libs safely.
try:
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# same embedding model as Assignment 8
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# where Assignment 8 wrote its outputs (Project/Preprocessing/processed-data)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Preprocessing" / "processed-data"


@dataclass
class RetrievedArticle:
    article_id: int
    title: str
    summary: str
    score: float           # weighted score used for ranking
    title_score: float     # similarity from the title-only index
    combined_score: float  # similarity from the title+summary index
    source: str = "semantic"  # "semantic" or "keyword" (fallback)


class ArticleRetriever:
    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        db_path: Optional[Path] = None,
        title_index_path: Optional[Path] = None,
        combined_index_path: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
        title_weight: float = 0.5,
    ) -> None:
        data_dir = Path(data_dir)
        self.db_path = Path(db_path) if db_path else data_dir / "wikihow.db"
        title_index_path = title_index_path or data_dir / "title_index.faiss"
        combined_index_path = combined_index_path or data_dir / "combined_index.faiss"

        for path in (self.db_path, title_index_path, combined_index_path):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Missing Assignment 8 artifact: {path}. "
                    "Run Preprocessing/wikihow_text_preprocessing.py first."
                )

        # default weighting between the two indices, overridable per query
        self.title_weight = float(title_weight)
        self.model = SentenceTransformer(model_name)
        self.title_index = faiss.read_index(str(title_index_path))
        self.combined_index = faiss.read_index(str(combined_index_path))
        # check_same_thread=False lets a Gradio worker thread reuse the connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _encode(self, query: str) -> np.ndarray:
        # encode + normalise the query the same way Assignment 8 did
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def _fetch_metadata(self, article_id: int) -> tuple[str, str]:
        # a FAISS position maps straight to the SQLite `id` (same row order at build time)
        row = self.conn.execute(
            "SELECT title, summary FROM articles WHERE id = ?", (int(article_id),)
        ).fetchone()
        if row is None:
            return "", ""
        title, summary = row
        return str(title or ""), str(summary or "")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        title_weight: Optional[float] = None,
        candidate_multiplier: int = 4,
    ) -> list[RetrievedArticle]:
        # search both indices and merge with score = w*title + (1-w)*combined
        w = self.title_weight if title_weight is None else float(title_weight)
        query_vec = self._encode(query)

        # pull a wider candidate pool from each index before merging
        pool = max(k * candidate_multiplier, k)
        title_scores, title_ids = self.title_index.search(query_vec, pool)
        combined_scores, combined_ids = self.combined_index.search(query_vec, pool)

        # collect per-article component scores keyed by article id
        scores: dict[int, dict[str, float]] = {}
        for score, idx in zip(title_scores[0], title_ids[0]):
            if idx == -1:
                continue
            scores.setdefault(int(idx), {"title": 0.0, "combined": 0.0})["title"] = float(score)
        for score, idx in zip(combined_scores[0], combined_ids[0]):
            if idx == -1:
                continue
            scores.setdefault(int(idx), {"title": 0.0, "combined": 0.0})["combined"] = float(score)

        ranked = sorted(
            scores.items(),
            key=lambda item: w * item[1]["title"] + (1 - w) * item[1]["combined"],
            reverse=True,
        )[:k]

        results: list[RetrievedArticle] = []
        for article_id, comp in ranked:
            title, summary = self._fetch_metadata(article_id)
            results.append(
                RetrievedArticle(
                    article_id=article_id,
                    title=title,
                    summary=summary,
                    score=w * comp["title"] + (1 - w) * comp["combined"],
                    title_score=comp["title"],
                    combined_score=comp["combined"],
                )
            )
        return results

    def keyword_search(self, query: str, k: int = 5) -> list[RetrievedArticle]:
        # fallback: plain LIKE match over the metadata
        like = f"%{query.strip()}%"
        rows = self.conn.execute(
            """
            SELECT id, title, summary FROM articles
            WHERE title LIKE ? OR summary LIKE ?
            LIMIT ?
            """,
            (like, like, k),
        ).fetchall()
        return [
            RetrievedArticle(
                article_id=int(row[0]),
                title=str(row[1] or ""),
                summary=str(row[2] or ""),
                score=0.0,
                title_score=0.0,
                combined_score=0.0,
                source="keyword",
            )
            for row in rows
        ]

    def retrieve_with_fallback(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.15,
        **kwargs,
    ) -> list[RetrievedArticle]:
        # use keyword search when semantic results are empty or too weak
        results = self.retrieve(query, k=k, **kwargs)
        if not results or results[0].score < min_score:
            keyword_results = self.keyword_search(query, k=k)
            if keyword_results:
                return keyword_results
        return results

    def format_context(
        self,
        articles: list[RetrievedArticle],
        max_chars_per_article: int = 500,
    ) -> str:
        # build the snippet that goes into the <|context|> block of the chatbot prompt.
        # summaries are truncated to keep the context within the token budget.
        if not articles:
            return ""
        lines = ["Relevant how-to articles:"]
        for i, art in enumerate(articles, start=1):
            summary = art.summary.strip().replace("\n", " ")
            if len(summary) > max_chars_per_article:
                summary = summary[:max_chars_per_article].rstrip() + "…"
            lines.append(f"{i}. {art.title.strip()}: {summary}")
        return "\n".join(lines)

    def close(self) -> None:
        self.conn.close()


def _demo() -> None:
    parser = argparse.ArgumentParser(description="ArticleRetriever quick test")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--query", type=str, default="how to clean a saddle?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--title-weight", type=float, default=0.5)
    args = parser.parse_args()

    retriever = ArticleRetriever(data_dir=args.data_dir, title_weight=args.title_weight)
    print(f"Query: {args.query}\n")
    for rank, art in enumerate(retriever.retrieve_with_fallback(args.query, k=args.k), start=1):
        print(
            f"{rank}. [{art.source}] score={art.score:.4f} "
            f"(title={art.title_score:.3f}, combined={art.combined_score:.3f})"
        )
        print(f"   {art.title}")
        print(f"   {art.summary[:160]}\n")

    print("---- Context block ----")
    print(retriever.format_context(retriever.retrieve(args.query, k=args.k)))
    retriever.close()


if __name__ == "__main__":
    _demo()
