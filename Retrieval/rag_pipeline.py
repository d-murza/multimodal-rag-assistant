# Assignment 10 - Task 3: end-to-end RAG pipeline.
# query -> article retrieval -> prompt/context -> TinyLlama generation
#       -> parse steps -> CLIP video retrieval per step.
from __future__ import annotations

# load pyarrow before torch/transformers to avoid the Windows import segfault
# (same reason as in article_retriever.py)
try:
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import pipeline

from article_retriever import ArticleRetriever, RetrievedArticle, DEFAULT_DATA_DIR
from video_retriever import VideoRetriever, RetrievedClip

# reuse the Assignment 9 chatbot core (prompt construction, validation, system prompt)
CHATBOT_DIR = Path(__file__).resolve().parent.parent / "Chatbot"
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))
from model_setup import system_prompt, prompt_construction, output_validation  # noqa: E402

DEFAULT_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_steps(text: str) -> list[str]:
    # pull out numbered steps ("1. ...", "2) ..."); each one becomes a video query.
    # if there is no list, fall back to the whole text as a single step.
    steps = [m.strip() for m in re.findall(r"(?m)^\s*\d+[\.\)]\s*(.+)$", text)]
    if steps:
        return steps
    cleaned = text.strip()
    return [cleaned] if cleaned else []


class ChatBot:
    # Assignment 9 TinyLlama model wrapped for single-turn RAG generation
    def __init__(
        self,
        model_name: str = DEFAULT_LLM,
        max_new_tokens: int = 256,
        max_context_tokens: int = 2048,
    ) -> None:

        device, dtype = "cpu", torch.float32

        self.pipe = pipeline("text-generation", model=model_name, dtype=dtype, device=device)
        self.tokenizer = self.pipe.tokenizer
        self.max_new_tokens = max_new_tokens
        # leave room for the generated tokens (same budgeting as Assignment 9)
        self.max_tokens = max_context_tokens - max_new_tokens

    def generate(
        self,
        context_block: str,
        conversation_history: list[dict],
    ) -> tuple[str, str, str, int]:
        # returns (prompt, raw_response, validated_response, prompt_token_count)
        history = list(conversation_history)
        prompt = prompt_construction(system_prompt, context_block, history)

        # drop the oldest turns until the prompt fits the budget (Assignment 9 strategy)
        token_count = len(self.tokenizer.encode(prompt))
        while token_count > self.max_tokens and len(history) > 1:
            history.pop(0)
            prompt = prompt_construction(system_prompt, context_block, history)
            token_count = len(self.tokenizer.encode(prompt))

        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        raw_response = outputs[0]["generated_text"][len(prompt):].strip()
        validated = output_validation(raw_response)
        return prompt, raw_response, validated, token_count


@dataclass
class RAGResult:
    query: str
    articles: list[RetrievedArticle]
    context_block: str
    prompt: str
    response: str                       # validated response ("Error" if rejected)
    prompt_token_count: int
    max_tokens: int
    steps: list[str] = field(default_factory=list)
    step_clips: list[tuple[str, list[RetrievedClip]]] = field(default_factory=list)

    def pretty_print(self) -> None:
        print(f"\n=== Query: {self.query} ===")
        print("\n[Retrieved articles] (context injected into <|context|>)")
        for i, art in enumerate(self.articles, start=1):
            print(f"  {i}. [{art.source}] score={art.score:.3f} :: {art.title}")
        print(f"\n[Token budget] {self.prompt_token_count} / {self.max_tokens} usable")
        print(f"\n[Generated response]\n{self.response}")
        print("\n[Video clips per generated step]")
        for step, clips in self.step_clips:
            print(f"  Step: {step}")
            for clip in clips:
                print(
                    f"     -> score={clip.score:.3f} | video_id={clip.video_id} | "
                    f"{clip.segment} | task={clip.task_name}"
                )


class RAGPipeline:
    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        article_k: int = 5,
        video_k_per_step: int = 1,
        title_weight: float = 0.5,
        load_chatbot: bool = True,
        llm_name: str = DEFAULT_LLM,
    ) -> None:
        self.article_k = article_k
        self.video_k_per_step = video_k_per_step
        self.article_retriever = ArticleRetriever(data_dir=data_dir, title_weight=title_weight)
        self.video_retriever = VideoRetriever(data_dir=data_dir)
        self.chatbot: Optional[ChatBot] = ChatBot(model_name=llm_name) if load_chatbot else None

    def run(self, query: str, conversation_history: Optional[list[dict]] = None) -> RAGResult:
        # 1. retrieve articles (semantic, keyword fallback)
        articles = self.article_retriever.retrieve_with_fallback(query, k=self.article_k)

        # 2. build the <|context|> block from titles + summaries
        context_block = self.article_retriever.format_context(articles)

        # 3. construct prompt and generate the answer
        history = list(conversation_history or [])
        history.append({"role": "user", "content": query})

        if self.chatbot is None:
            raise RuntimeError("RAGPipeline was created with load_chatbot=False.")
        prompt, raw_response, response, token_count = self.chatbot.generate(context_block, history)

        # 4. split the answer into steps (use raw text if validation rejected it)
        source_text = raw_response if response == "Error" else response
        steps = parse_steps(source_text)

        # 5. retrieve one video clip per generated step
        step_clips = self.video_retriever.retrieve_for_steps(steps, k_per_step=self.video_k_per_step)

        return RAGResult(
            query=query,
            articles=articles,
            context_block=context_block,
            prompt=prompt,
            response=response,
            prompt_token_count=token_count,
            max_tokens=self.chatbot.max_tokens,
            steps=steps,
            step_clips=step_clips,
        )

    def close(self) -> None:
        self.article_retriever.close()
        self.video_retriever.close()


def _demo() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG pipeline demo")
    parser.add_argument("--query", type=str, default="How to change a tire?")
    parser.add_argument("--article-k", type=int, default=5)
    parser.add_argument("--video-k-per-step", type=int, default=1)
    parser.add_argument("--title-weight", type=float, default=0.5)
    args = parser.parse_args()

    pipeline_obj = RAGPipeline(
        article_k=args.article_k,
        video_k_per_step=args.video_k_per_step,
        title_weight=args.title_weight,
    )
    result = pipeline_obj.run(args.query)
    result.pretty_print()
    pipeline_obj.close()


if __name__ == "__main__":
    _demo()
