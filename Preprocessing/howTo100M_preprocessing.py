from __future__ import annotations

import argparse
import csv
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import cv2

# data obj

DATA_DIR = "data/howto100m_bundle"

@dataclass(frozen=True)
class ClipRecord:
    clip_id: int
    video_id: str
    task_name: str
    video_path: str
    clip_index: int
    start_sec: float
    end_sec: float
    frame_count: int


# helpers

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()

# csv load

def load_task_video_selection(task_ids_csv, available_videos_csv):
    task_df = pd.read_csv(task_ids_csv,
        sep='\t',
        header=None,
        names=['task_id', 'task_name'],
    )
    videos_df = pd.read_csv(available_videos_csv)

    task_df["task_id"] = task_df["task_id"].astype(str)
    videos_df["task_id"] = videos_df["task_id"].astype(str)

    merged = videos_df.merge(
        task_df,
        on="task_id",
        how="inner"
    )

    return merged

# get video path

def resolve_video_path(row: pd.Series, videos_root: Path) -> Path:
    # just in case normalization of a path (strip, etc.)
    raw_path = normalize_text(row.get("video_path", ""))
    if raw_path:
        p = Path(raw_path)
        if p.is_absolute() and p.exists():
            return p
        candidate = videos_root / p
        if candidate.exists():
            return candidate
        if candidate.suffix == "":
            mp4_candidate = candidate.with_suffix(".mp4")
            if mp4_candidate.exists():
                return mp4_candidate

    fallback = videos_root / f"{normalize_text(row['video_id'])}.mp4"
    if fallback.exists():
        return fallback

# generate the clips out of video

def get_video_duration_and_fps(video_path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps and fps > 0 else 0.0
    cap.release()
    return duration_sec, fps if fps and fps > 0 else 0.0, frame_count


def generate_sliding_window_clips(
    video_id: str,
    video_path: Path,
    window_sec: float = 8.0,
    stride_sec: float = 4.0,
    min_clip_sec: float = 2.0,
) -> list[dict]:
    duration_sec, _, _ = get_video_duration_and_fps(video_path)
    if duration_sec <= 0:
        return []

    if duration_sec <= window_sec:
        return [
            {
                "video_id": video_id,
                "video_path": str(video_path),
                "clip_index": 0,
                "start_sec": 0.0,
                "end_sec": float(duration_sec),
            }
        ]

    clips: list[dict] = []
    clip_index = 0
    start = 0.0
    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        if end - start >= min_clip_sec:
            clips.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "clip_index": clip_index,
                    "start_sec": float(start),
                    "end_sec": float(end),
                }
            )
            clip_index += 1
        if end >= duration_sec:
            break
        start += stride_sec

    return clips


# frame sampling and embedding


def sample_frame_times(start_sec: float, end_sec: float, num_frames: int) -> list[float]:
    if num_frames <= 1:
        return [(start_sec + end_sec) / 2.0]
    span = max(end_sec - start_sec, 1e-6)
    return [start_sec + span * (i + 0.5) / num_frames for i in range(num_frames)]


def read_frame_at_time(cap: cv2.VideoCapture, time_sec: float) -> Optional[Image.Image]:

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# ive found that we can use an attribute instead of "with torch.no_grad():" (looks cleaner)\
# not used anymore - rewrote into more efficient embed_video_clips() that decodes sequentially and batches the CLIP forward pass
@torch.no_grad()
def embed_clip_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    frames_per_clip: int = 4,
) -> np.ndarray:

    times = sample_frame_times(start_sec, end_sec, frames_per_clip)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((512,), dtype=np.float32)

    images: list[Image.Image] = []
    for t in times:
        img = read_frame_at_time(cap, t)
        if img is not None:
            images.append(img)
    cap.release()

    if not images:
        return np.zeros((512,), dtype=np.float32)

    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = normalize_features(image_features)
    clip_feature = image_features.mean(dim=0)
    clip_feature = torch.nn.functional.normalize(clip_feature, dim=0)
    return clip_feature.cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_video_clips(
    video_path: Path,
    clips: list[dict],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    frames_per_clip: int = 4,
    batch_size: int = 64,
) -> list[np.ndarray]:
    def zero() -> np.ndarray:
        return np.zeros((512,), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [zero() for _ in clips]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # GUard check

    # 1) map each clip to the frame indices it samples (same timestamps as before).
    clip_frame_indices: list[list[int]] = []
    needed: set[int] = set()
    for clip in clips:
        times = sample_frame_times(float(clip["start_sec"]), float(clip["end_sec"]), frames_per_clip)
        idxs = [max(0, int(round(t * fps))) for t in times]
        clip_frame_indices.append(idxs)
        needed.update(idxs)
    max_idx = max(needed) if needed else -1

    # 2) sequential decode. grab() advances+decodes cheaply; we only retrieve()+convert
    #    the frames we actually need. Frames are embedded in batches and then freed, so
    #    memory stays bounded (we keep the tiny 512-d features, not the images).
    feat_by_index: dict[int, torch.Tensor] = {}
    buf_imgs: list[Image.Image] = []
    buf_idxs: list[int] = []

    def flush() -> None:
        if not buf_imgs:
            return
        inputs = processor(images=buf_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        feats = normalize_features(model.get_image_features(pixel_values=pixel_values)).cpu()
        for j, idx in enumerate(buf_idxs):
            feat_by_index[idx] = feats[j]
        buf_imgs.clear()
        buf_idxs.clear()

    frame_idx = 0
    while frame_idx <= max_idx:
        if not cap.grab(): # decode&advance without copying pixels
            break
        if frame_idx in needed:
            ok, frame = cap.retrieve() # pull pxels only for frames we need
            if ok and frame is not None:
                buf_imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                buf_idxs.append(frame_idx)
                if len(buf_imgs) >= batch_size:
                    flush()
        frame_idx += 1
    flush()
    cap.release()

    # 3) per clip: average its frames' features and normalise (same as embed_clip_frames).
    vectors: list[np.ndarray] = []
    for idxs in clip_frame_indices:
        rows = [feat_by_index[i] for i in idxs if i in feat_by_index]
        if not rows:
            vectors.append(zero())
            continue
        clip_feature = torch.nn.functional.normalize(torch.stack(rows).mean(dim=0), dim=0)
        vectors.append(clip_feature.numpy().astype(np.float32))
    return vectors


def normalize_features(features):
    if hasattr(features, "image_embeds"):
        features = features.image_embeds
    elif hasattr(features, "text_embeds"):
        features = features.text_embeds
    elif hasattr(features, "pooler_output"):
        features = features.pooler_output

    if not torch.is_tensor(features):
        features = torch.as_tensor(features)

    return features / features.norm(p=2, dim=-1, keepdim=True)


@torch.no_grad()
def embed_text(
    query: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    text_features = normalize_features(text_features)
    return text_features.cpu().numpy().astype(np.float32)


# -----------------------------
# Storage: SQLite + FAISS
# -----------------------------


def init_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            clip_id INTEGER PRIMARY KEY,
            video_id TEXT NOT NULL,
            task_name TEXT,
            video_path TEXT NOT NULL,
            clip_index INTEGER NOT NULL,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            frame_count INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id)")
    conn.commit()
    return conn


def next_clip_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(clip_id), 0) FROM clips").fetchone()
    return int(row[0]) + 1


def insert_clips(conn: sqlite3.Connection, records: list[ClipRecord]) -> None:
    rows = [
        (
            rec.clip_id,
            rec.video_id,
            rec.task_name,
            rec.video_path,
            rec.clip_index,
            rec.start_sec,
            rec.end_sec,
            rec.frame_count,
        )
        for rec in records
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO clips
        (clip_id, video_id, task_name, video_path, clip_index, start_sec, end_sec, frame_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def make_faiss_index(embedding_dim: int = 512) -> faiss.IndexIDMap2:
    base = faiss.IndexFlatIP(embedding_dim)
    return faiss.IndexIDMap2(base)


def save_index(index: faiss.IndexIDMap2, index_path: Path) -> None:
    faiss.write_index(index, str(index_path))


# -----------------------------
# Retrieval and verification
# -----------------------------


def fetch_clip_metadata(conn: sqlite3.Connection, clip_id: int) -> dict:
    row = conn.execute(
        """
        SELECT clip_id, video_id, task_name, video_path, clip_index, start_sec, end_sec, frame_count
        FROM clips
        WHERE clip_id = ?
        """,
        (clip_id,),
    ).fetchone()

    keys = ["clip_id", "video_id", "task_name", "video_path", "clip_index", "start_sec", "end_sec", "frame_count"]
    return dict(zip(keys, row))

# def play_clip(video_path, start_sec, end_sec, title="Retrieved Clip"):
#     cap = cv2.VideoCapture(str(video_path))

#     if not cap.isOpened():
#         print(f"Could not open: {video_path}")
#         return

#     cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     delay = max(1, int(1000 / fps)) if fps > 0 else 25

#     while True:
#         current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

#         if current_sec >= end_sec:
#             break

#         ret, frame = cap.read()
#         if not ret:
#             break

#         cv2.imshow(title, frame)

#         key = cv2.waitKey(delay)

#         # Close clip immediately with Q or ESC
#         if key == ord("q") or key == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def play_retrieved_clips(retrieved_clips):
#     """
#     retrieved_clips = [
#         {
#             "video_path": "...",
#             "start_sec": 168,
#             "end_sec": 176
#         },
#         ...
#     ]
#     """

#     for i, clip in enumerate(retrieved_clips, start=1):
#         print(
#             f"\nPlaying clip {i}/{len(retrieved_clips)} "
#             f"({clip['start_sec']:.1f}s - {clip['end_sec']:.1f}s)"
#         )

#         play_clip(
#             clip["video_path"],
#             clip["start_sec"],
#             clip["end_sec"],
#             title=f"Result {i}"
#         )

#         print("Clip closed. Opening next result...")


def retrieval_test(
    query: str,
    index: faiss.IndexIDMap2,
    conn: sqlite3.Connection,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    top_k: int = 5,
) -> None:
    query_vec = embed_text(query, model, processor, device)
    scores, ids = index.search(query_vec, top_k)

    # retrieved_clips = []

    print(f"\nQuery: {query}\n")
    for rank, (score, clip_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if clip_id == -1:
            continue
        meta = fetch_clip_metadata(conn, int(clip_id))
        print(
            f"{rank}. score={score:.4f} | video_id={meta['video_id']} | "
            f"clip_index={meta['clip_index']} | {meta['start_sec']:.1f}s-{meta['end_sec']:.1f}s"
        )

        # retrieved_clips.append({
        #     "video_path": meta["video_path"],
        #     "start_sec": meta["start_sec"],
        #     "end_sec": meta["end_sec"]
        # })
    
    # play_retrieved_clips(retrieved_clips)


# pipeline

def preprocess_videos(
    task_ids_csv: Path,
    available_videos_csv: Path,
    videos_root: Path,
    db_path: Path,
    index_path: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    window_sec: float = 8.0,
    stride_sec: float = 4.0,
    min_clip_sec: float = 2.0,
    frames_per_clip: int = 4,
    batch_size: int = 64,
    limit_videos: Optional[int] = None,
) -> tuple[sqlite3.Connection, faiss.IndexIDMap2, CLIPModel, CLIPProcessor, torch.device]:
    videos_root = Path(videos_root)

    merged = load_task_video_selection(task_ids_csv, available_videos_csv)
    if limit_videos is not None:
        merged = merged.head(limit_videos).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    conn = init_database(db_path)
    index = make_faiss_index(embedding_dim=512)
    clip_id_cursor = next_clip_id(conn)

    # process each video independently and insert them into database to minimise ram usage
    for row_idx, row in merged.iterrows():
        video_id = normalize_text(row["video_id"])
        video_path = resolve_video_path(row, videos_root)

        clips = generate_sliding_window_clips(
            video_id=video_id,
            video_path=video_path,
            window_sec=window_sec,
            stride_sec=stride_sec,
            min_clip_sec=min_clip_sec,
        )

        print(f"[video {row_idx + 1}/{len(merged)}] {video_id} -> {len(clips)} clips")

        if not clips:
            continue

        # embed every clip of this video with a single sequential decode + batched CLIP.
        clip_vectors = embed_video_clips(
            video_path=video_path,
            clips=clips,
            model=model,
            processor=processor,
            device=device,
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
        )

        clip_records: list[ClipRecord] = []
        clip_vector_ids: list[int] = []

        for clip in clips:
            clip_id = clip_id_cursor
            clip_id_cursor += 1

            clip_records.append(
                ClipRecord(
                    clip_id=clip_id,
                    video_id=video_id,
                    task_name=str(row["task_name"]),
                    video_path=str(video_path),
                    clip_index=int(clip["clip_index"]),
                    start_sec=float(clip["start_sec"]),
                    end_sec=float(clip["end_sec"]),
                    frame_count=frames_per_clip,
                )
            )
            clip_vector_ids.append(clip_id)

        insert_clips(conn, clip_records)

        vectors = np.vstack(clip_vectors).astype(np.float32)
        faiss.normalize_L2(vectors)
        ids = np.array(clip_vector_ids, dtype=np.int64)
        index.add_with_ids(vectors, ids)

        print(f"  added {len(clips)} clips | total indexed clips: {index.ntotal}")

    save_index(index, index_path)
    return conn, index, model, processor, device


# i decided to create a arg parser so that we are flexible with the input files
# to run on Windows Powershell: 
# python howTo100M_preprocessing.py `
#  --limit-videos 5 `
#  --query "how to boil a water"

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HowTo100M Task #2 preprocessing pipeline")
    parser.add_argument("--db-path", type=Path, default=Path("processed-data/howto100m_clips.sqlite3"))
    parser.add_argument("--index-path", type=Path, default=Path("processed-data/howto100m_clips.faiss"))
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Frozen CLIP model name (ViT-B/32 compatible)",
    )
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=4.0)
    parser.add_argument("--min-clip-sec", type=float, default=2.0)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of frames per CLIP forward pass (raise to use more GPU)",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Process only the first N videos for debugging (e.g. 2-5)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional retrieval test query, e.g. 'how to cut onions'",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser

def main() -> None:
    args = build_argparser().parse_args()

    conn, index, model, processor, device = preprocess_videos(
        task_ids_csv=os.path.join(DATA_DIR, "task_ids.csv"),
        available_videos_csv=os.path.join(DATA_DIR, "available_videos.csv"),
        videos_root=os.path.join(DATA_DIR, "videos"),
        db_path=args.db_path,
        index_path=args.index_path,
        model_name=args.model_name,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        min_clip_sec=args.min_clip_sec,
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        limit_videos=args.limit_videos,
    )

    total_clips = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    print(f"\nDatabase contains {total_clips} clips.")
    print(f"FAISS index size: {index.ntotal}")

    if args.query:
        retrieval_test(
            query=args.query,
            index=index,
            conn=conn,
            model=model,
            processor=processor,
            device=device,
            top_k=args.top_k,
        )

    conn.close()

if __name__ == "__main__":
    main()
