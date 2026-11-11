# Text-to-clip retrieval over the HowTo100M CLIP index
# built during preprocessing (howto100m_clips.faiss + howto100m_clips.sqlite3).
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

# reuse the CLIP text encoder and metadata helper from the preprocessing script
PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "Preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))
from howTo100M_preprocessing import embed_text, fetch_clip_metadata  # noqa: E402

# same CLIP model and output folder as preprocessing
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Preprocessing" / "processed-data"


@dataclass
class RetrievedClip:
    clip_id: int
    video_id: str
    task_name: str
    clip_index: int
    start_sec: float
    end_sec: float
    score: float
    video_path: str = ""

    @property
    def segment(self) -> str:
        return f"{self.start_sec:.1f}s-{self.end_sec:.1f}s"


class VideoRetriever:
    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        data_dir = Path(data_dir)
        self.index_path = Path(index_path) if index_path else data_dir / "howto100m_clips.faiss"
        self.db_path = Path(db_path) if db_path else data_dir / "howto100m_clips.sqlite3"

        for path in (self.index_path, self.db_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing preprocessing artifact: {path}. "
                    "Run Preprocessing/howTo100M_preprocessing.py first."
                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # IndexIDMap2 returns the stored clip_id, not a positional index
        self.index = faiss.read_index(str(self.index_path))
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedClip]:
        # clip vectors are L2-normalised, so inner product == cosine similarity
        query_vec = embed_text(query, self.model, self.processor, self.device)

        scores, clip_ids = self.index.search(query_vec, k)

        results: list[RetrievedClip] = []
        for score, clip_id in zip(scores[0], clip_ids[0]):
            if clip_id == -1:
                continue
            meta = fetch_clip_metadata(self.conn, int(clip_id))
            results.append(
                RetrievedClip(
                    clip_id=int(meta["clip_id"]),
                    video_id=str(meta["video_id"]),
                    task_name=str(meta.get("task_name") or ""),
                    clip_index=int(meta["clip_index"]),
                    start_sec=float(meta["start_sec"]),
                    end_sec=float(meta["end_sec"]),
                    score=float(score),
                    video_path=str(meta.get("video_path") or ""),
                )
            )
        return results

    def retrieve_for_steps(
        self,
        steps: list[str],
        k_per_step: int = 1,
    ) -> list[tuple[str, list[RetrievedClip]]]:
        # one clip per generated instruction step; used by the RAG pipeline
        return [(step, self.retrieve(step, k=k_per_step)) for step in steps if step.strip()]

    def close(self) -> None:
        self.conn.close()


def _demo() -> None:
    parser = argparse.ArgumentParser(description="VideoRetriever quick test")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--query", type=str, default="how to boil water")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    retriever = VideoRetriever(data_dir=args.data_dir)
    print(f"Query: {args.query}\n")
    for rank, clip in enumerate(retriever.retrieve(args.query, k=args.k), start=1):
        print(
            f"{rank}. score={clip.score:.4f} | video_id={clip.video_id} | "
            f"clip_index={clip.clip_index} | {clip.segment} | task={clip.task_name}"
        )
    retriever.close()


if __name__ == "__main__":
    _demo()
