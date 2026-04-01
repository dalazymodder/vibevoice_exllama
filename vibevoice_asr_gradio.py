#!/usr/bin/env python3
"""
VibeVoice-ASR + ExLlamaV3 — Gradio Web UI
==========================================
A web interface for the VibeVoice-ASR speech-to-text pipeline with
ExLlamaV3 quantized inference.

Usage:
  python vibevoice_asr_gradio.py
  python vibevoice_asr_gradio.py --listen            # bind to 0.0.0.0
  python vibevoice_asr_gradio.py --port 7861         # custom port
  python vibevoice_asr_gradio.py --llm_dir /my/llm   # override model paths

All model arguments have the same defaults as the CLI script and can be
overridden via command-line flags passed to this launcher.
"""

import argparse
import json
import os
import sys
import time
import tempfile

import gradio as gr

# Import the pipeline from the inference script (must be on PYTHONPATH or same dir)
from vibevoice_asr_exl3_inference import VibeVoiceASRPipeline


# ---------------------------------------------------------------------------
# Global pipeline (loaded once at startup)
# ---------------------------------------------------------------------------
PIPELINE: VibeVoiceASRPipeline = None  # type: ignore


def load_pipeline(encoders_dir, connectors_dir, llm_dir, device, cache_tokens):
    """Instantiate the pipeline (heavy — runs once)."""
    global PIPELINE
    if PIPELINE is None:
        print("\n⏳ Loading VibeVoice-ASR pipeline (this may take a minute)…")
        PIPELINE = VibeVoiceASRPipeline(
            encoders_dir=encoders_dir,
            connectors_dir=connectors_dir,
            llm_dir=llm_dir,
            device=device,
            cache_tokens=cache_tokens,
        )
        print("✅ Pipeline ready!\n")
    return PIPELINE


# ---------------------------------------------------------------------------
# Core transcription function wired to Gradio
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    context: str,
    max_new_tokens: int,
    temperature: float,
):
    """Run the ASR pipeline on an uploaded/recorded audio file."""
    if audio_path is None or not os.path.exists(audio_path):
        return (
            "⚠️ No audio provided. Upload a file or record via microphone.",
            "",
            "",
        )

    pipeline = PIPELINE
    if pipeline is None:
        return ("❌ Pipeline not loaded.", "", "")

    try:
        result = pipeline.transcribe(
            audio_path=audio_path,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            context_info=context.strip() if context and context.strip() else None,
        )
    except Exception as e:
        return (f"❌ Error during transcription:\n{e}", "", "")

    # ── Format the segments into readable text ──
    lines = []
    segments = result.get("segments", [])
    if segments:
        for seg in segments:
            start = seg.get("Start time", seg.get("Start", seg.get("start", "?")))
            end = seg.get("End time", seg.get("End", seg.get("end", "?")))
            speaker = seg.get("Speaker ID", seg.get("Speaker", seg.get("speaker", "?")))
            content = seg.get("Content", seg.get("content", seg.get("text", "")))
            lines.append(f"[{start} → {end}]  Speaker {speaker}:  {content}")
        formatted = "\n".join(lines)
    else:
        formatted = result.get("raw_text", "(no output)")

    # ── Stats summary ──
    dur = result.get("audio_duration", 0)
    stats = (
        f"Audio duration:   {dur:.1f}s\n"
        f"Encoder:          {result.get('encoder_ms', 0):.0f} ms\n"
        f"Generation:       {result.get('generation_ms', 0):.0f} ms  "
        f"({result.get('num_tokens', 0)} tokens, "
        f"{result.get('tokens_per_second', 0):.1f} tok/s)\n"
        f"Total pipeline:   {result.get('total_ms', 0):.0f} ms\n"
        f"Realtime factor:  {result.get('realtime_factor', 0):.2f}x"
    )

    # ── JSON blob ──
    json_out = json.dumps(result, indent=2, ensure_ascii=False)

    return formatted, stats, json_out


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Overall tweaks ── */
.gradio-container {
    max-width: 960px !important;
    margin: auto;
}
#title-row {
    text-align: center;
    margin-bottom: 0.25rem;
}
#title-row h1 {
    font-size: 1.8rem;
    margin-bottom: 0.1rem;
}
#title-row p {
    opacity: 0.65;
    font-size: 0.95rem;
}
/* transcript box */
#transcript-box textarea {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}
#stats-box textarea {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.85rem !important;
}
"""

def build_ui():
    with gr.Blocks(
        title="VibeVoice-ASR",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:

        # ── Header ──
        with gr.Row(elem_id="title-row"):
            gr.Markdown(
                "# 🎙️ VibeVoice-ASR\n"
                "Speech-to-text with ExLlamaV3 quantized inference  ·  "
                "Upload audio or record from your microphone"
            )

        # ── Main layout ──
        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio input",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                context_input = gr.Textbox(
                    label="Context / hotwords",
                    placeholder="e.g. Alice, Bob, Q3 earnings",
                    lines=1,
                )
                with gr.Accordion("Advanced settings", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=16384,
                        value=8192,
                        step=256,
                        label="Max new tokens",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=0.0,
                        step=0.05,
                        label="Temperature (0 = greedy)",
                    )

                transcribe_btn = gr.Button(
                    "Transcribe",
                    variant="primary",
                    size="lg",
                )

            # Right column: outputs
            with gr.Column(scale=2):
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=16,
                    max_lines=40,
                    elem_id="transcript-box",
                )
                stats_output = gr.Textbox(
                    label="Performance",
                    lines=5,
                    interactive=False,
                    elem_id="stats-box",
                )
                with gr.Accordion("Raw JSON", open=False):
                    json_output = gr.Code(
                        label="Full result",
                        language="json",
                        lines=12,
                    )

        # ── Wiring ──
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, context_input, max_tokens_slider, temperature_slider],
            outputs=[transcript_output, stats_output, json_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="VibeVoice-ASR Gradio UI")

    # Model paths (same defaults as your CLI command)
    p.add_argument("--encoders_dir",   default="vibevoice_asr_exllama_q8/encoders",    help="Path to encoder weights")
    p.add_argument("--connectors_dir", default="vibevoice_asr_exllama_q8/connectors",  help="Path to connector weights")
    p.add_argument("--llm_dir",        default="vibevoice_asr_exllama_q8/vibex",       help="Path to ExLlamaV3 LLM dir")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--cache_tokens",   type=int, default=32768)

    # Server options
    p.add_argument("--listen",  action="store_true", help="Bind to 0.0.0.0 (LAN access)")
    p.add_argument("--port",    type=int, default=7860)
    p.add_argument("--share",   action="store_true",  help="Create a Gradio public link")
    p.add_argument("--auth",    type=str, default=None,
                   help="Optional user:pass for basic auth (e.g. admin:secret)")

    args = p.parse_args()

    # Load the heavy pipeline once before launching
    load_pipeline(
        encoders_dir=args.encoders_dir,
        connectors_dir=args.connectors_dir,
        llm_dir=args.llm_dir,
        device=args.device,
        cache_tokens=args.cache_tokens,
    )

    demo = build_ui()

    auth = None
    if args.auth:
        user, pw = args.auth.split(":", 1)
        auth = (user, pw)

    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        auth=auth,
    )


if __name__ == "__main__":
    main()
