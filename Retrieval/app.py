# Gradio interface for the RAG pipeline.
# Left: chat. Right: retrieved articles + video clips (click a clip to play it
# at its own timestamp). Bottom: a collapsible inspection panel.
from __future__ import annotations

# load pyarrow before the heavy ML imports to avoid the Windows segfault
# (same reason as in article_retriever.py)
try:
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import urllib.parse
from pathlib import Path

import gradio as gr

from rag_pipeline import RAGPipeline
from model_setup import system_prompt, overall_max_tokens

# where the HowTo100M mp4 files live (used to play a selected clip)
PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "Preprocessing"
VIDEOS_DIR = (PREPROCESSING_DIR / "data" / "howto100m_bundle" / "videos").resolve()

# Gradio serves local files under this route (confirmed for this version)
FILE_ROUTE = "/gradio_api/file="
PLACEHOLDER_HTML = "<p style='color:#888'>Click a clip in the table to play it here.</p>"

# building the pipeline loads three models, so we do it once and reuse it
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def resolve_video_file(clip) -> str | None:
    # find the mp4 for a clip: try the stored path, then rebuild it from the video id
    stored = Path(clip.video_path) if clip.video_path else None
    if stored and stored.exists():
        return str(stored.resolve())
    if stored and (PREPROCESSING_DIR / stored).exists():
        return str((PREPROCESSING_DIR / stored).resolve())
    guess = VIDEOS_DIR / f"{clip.video_id}.mp4"
    return str(guess) if guess.exists() else None


def clip_player_html(meta: dict) -> str:
    # play the original mp4 but use an HTML5 media fragment (#t=start,end) so it
    # starts at the clip's timestamp; add a caption saying which clip it is.
    src = FILE_ROUTE + urllib.parse.quote(Path(meta["path"]).as_posix())
    src += f"#t={meta['start']:.1f},{meta['end']:.1f}"
    caption = (
        f"&#9654; <b>{meta['video_id']}</b> &nbsp;&middot;&nbsp; {meta['segment']} "
        f"&nbsp;&middot;&nbsp; {meta['task']} &nbsp;&middot;&nbsp; score {meta['score']:.3f}"
    )
    return (
        f"<div><p style='margin:0 0 6px'>{caption}</p>"
        f"<video src='{src}' controls autoplay "
        f"style='width:100%;max-height:340px;border-radius:8px'></video></div>"
    )


def respond(message: str, chat_messages: list[dict], history_state: list[dict]):
    # run one RAG turn and refresh every panel
    message = (message or "").strip()
    if not message:
        return chat_messages, history_state, [], [], [], "0 / 0", [], "", PLACEHOLDER_HTML

    result = get_pipeline().run(message, conversation_history=history_state)

    display_response = (
        result.response
        if result.response != "Error"
        else "(The generated response was rejected by output validation. Please try again.)"
    )

    # add this turn to both the visible chat and the model-facing history
    turn = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": display_response},
    ]
    chat_messages = chat_messages + turn
    history_state = history_state + turn

    # articles table: title | score | source
    article_rows = [[art.title, round(art.score, 3), art.source] for art in result.articles]

    # videos table + full clip metadata (kept in a State so a click can play the clip)
    video_rows: list[list] = []
    clip_metas: list[dict] = []
    for step, clips in result.step_clips:
        for clip in clips:
            video_rows.append([step, clip.video_id, clip.segment, round(clip.score, 3), clip.task_name])
            clip_metas.append(
                {
                    "path": resolve_video_file(clip),
                    "start": clip.start_sec,
                    "end": clip.end_sec,
                    "video_id": clip.video_id,
                    "segment": clip.segment,
                    "task": clip.task_name,
                    "score": clip.score,
                }
            )

    # inspection panel
    token_budget = f"{result.prompt_token_count} / {overall_max_tokens} tokens used"
    vector_rows = [
        [f"Text #{i}", art.title[:40], round(art.score, 3)]
        for i, art in enumerate(result.articles, start=1)
    ]
    vector_rows += [
        [f"Video (step {i})", clip.video_id, round(clip.score, 3)]
        for i, (_step, clips) in enumerate(result.step_clips, start=1)
        for clip in clips
    ]
    injected_context = result.context_block or "(no context injected)"

    return (
        chat_messages,
        history_state,
        article_rows,
        video_rows,
        clip_metas,
        token_budget,
        vector_rows,
        injected_context,
        PLACEHOLDER_HTML,
    )


def play_selected(evt: gr.SelectData, clip_metas: list):
    # play the clip whose row was clicked in the videos table
    if clip_metas and evt.index is not None:
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if 0 <= row < len(clip_metas):
            meta = clip_metas[row]
            if meta["path"]:
                return clip_player_html(meta)
            return f"<p style='color:#c00'>Video file not found for {meta['video_id']}.</p>"
    return gr.update()


def build_ui() -> gr.Blocks:
    css = """
                    #videos-table td {
                        max-width: 250px;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    }
                    """
    with gr.Blocks(title="Multimodal How-To RAG", css=css) as demo:
        gr.Markdown(
            "# Multimodal How-To RAG Assistant\n"
            "Ask a how-to question — the system retrieves articles, generates the steps, "
            "and finds a video clip for each step."
        )

        gr.Markdown(
            f"#### Current system prompt:\n {system_prompt}\n"
        )

        history_state = gr.State([])
        clips_state = gr.State([])

        with gr.Row(equal_height=False):
            # chat on the left
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460, label="Chat")
                with gr.Row():
                    msg = gr.Textbox(
                        scale=5,
                        show_label=False,
                        placeholder="How to change a tire?",
                        autofocus=True,
                    )
                    send = gr.Button("Send", scale=1, variant="primary")
                clear = gr.Button("Clear conversation", size="sm")

            # retrieval results on the right
            with gr.Column(scale=2):
                with gr.Tab("Articles"):
                    articles_table = gr.Dataframe(
                        headers=["title", "score", "source"],
                        column_widths=["60%", "20%", "20%"],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Videos"):
                    video_html = gr.HTML(PLACEHOLDER_HTML, label="Selected clip")
                    videos_table = gr.Dataframe(
                        headers=["step", "video_id", "segment", "score", "task"],
                        interactive=False,
                        wrap=True,
                        elem_id="videos-table",
                    )

        with gr.Accordion("System inspection", open=False):
            token_budget = gr.Textbox(label="Token budget", interactive=False)
            vector_scores = gr.Dataframe(
                headers=["item", "ref", "score"],
                label="Similarity scores",
                interactive=False,
                wrap=True,
            )
            injected_context = gr.Textbox(
                label="Injected context block", lines=8, interactive=False
            )

        outputs = [
            chatbot,
            history_state,
            articles_table,
            videos_table,
            clips_state,
            token_budget,
            vector_scores,
            injected_context,
            video_html,
        ]
        # submit on Enter or the Send button, then clear the textbox
        msg.submit(respond, [msg, chatbot, history_state], outputs).then(lambda: "", None, msg)
        send.click(respond, [msg, chatbot, history_state], outputs).then(lambda: "", None, msg)

        # clicking a row in the videos table plays that clip
        videos_table.select(play_selected, clips_state, video_html)

        clear.click(
            lambda: ([], [], [], [], [], "0 / 0", [], "", PLACEHOLDER_HTML, ""),
            None,
            outputs + [msg],
        )

    return demo


if __name__ == "__main__":
    # allowed_paths lets Gradio serve the local mp4 files to the video player
    build_ui().launch(theme=gr.themes.Soft(), allowed_paths=[str(VIDEOS_DIR)])
