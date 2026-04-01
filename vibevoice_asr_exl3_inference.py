#!/usr/bin/env python3
"""
VibeVoice-ASR + ExLlamaV3 Quantized Inference
==============================================
Runs Microsoft's VibeVoice-ASR speech-to-text model with the LLM backbone
quantized to 8-bit (EXL3 format) via ExLlamaV3. This cuts VRAM from ~17GB
to ~10GB and speeds up generation ~4.7x — with no quality loss.

Architecture:
  ┌─────────────┐   ┌───────────────┐   ┌──────────────────┐
  │  Audio file  │──▶│ Acoustic enc. │──▶│ Acoustic connect.│──┐
  │  (wav/mp3)   │   │ (bf16, ~1GB)  │   │ (bf16, ~50MB)    │  │  element-wise
  └─────────────┘   └───────────────┘   └──────────────────┘  ├─────add────┐
                    ┌───────────────┐   ┌──────────────────┐  │             │
                    │ Semantic enc. │──▶│ Semantic connect. │──┘             │
                    │ (bf16, ~0.9GB)│   │ (bf16, ~50MB)    │               ▼
                    └───────────────┘   └──────────────────┘     speech_features
                                                                  [B, T, 3584]
                                                                       │
  ┌──────────────────────────────────────────────────────────────┐      │
  │  ExLlamaV3 Qwen2-7B (EXL3 8bpw, ~7.1GB)                    │      │
  │  ┌────────────┐                                              │      │
  │  │ Embedding   │◀── text tokens (system prompt, user prompt) │      │
  │  │ module      │◀── speech tokens via MMEmbedding ───────────│──────┘
  │  └─────┬──────┘                                              │
  │        ▼                                                     │
  │  [28 transformer layers with paged KV cache]                 │
  │        ▼                                                     │
  │  Autoregressive token generation ──▶ JSON transcript         │
  └──────────────────────────────────────────────────────────────┘

Key technical details:
  - Speech embeddings are injected via ExLlamaV3's native MMEmbedding API
    (the same system used for Gemma3, Qwen2.5-VL, etc. vision models).
  - No monkey-patching or custom forward hooks required.
  - Cache MUST be created before model.load() — see ExLlamaASRGenerator.

Benchmarks (3min podcast, RTX 4090):
  ExLlamaV3 8bpw:   ~55 tok/s, ~10GB VRAM, ~20s total
  Transformers bf16: ~12 tok/s, ~17GB VRAM, ~92s total

Directory structure (after split + quantization):
  split/
  ├── encoders/
  │   ├── acoustic/          # Acoustic encoder weights + config
  │   └── semantic/          # Semantic encoder weights + config
  ├── connectors/            # Acoustic + semantic connector weights + config
  └── llm-exl3-8.0bpw/      # Quantized LLM (ExLlamaV3 format)

Requirements:
  pip install torch exllamav3 safetensors numpy soundfile librosa transformers
  pip install vibevoice  # (editable install of the VibeVoice-ASR repo)

Usage:
  python vibevoice_asr_exl3_inference.py --audio_file recording.wav
  python vibevoice_asr_exl3_inference.py --audio_file meeting.wav --context "Alice,Bob,Q3"
  python vibevoice_asr_exl3_inference.py --audio_file call.wav --output_json result.json
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


# ============================================================================
# Audio utilities
# ============================================================================

def load_audio(path: str, target_sr: int = 24000) -> np.ndarray:
    """Load an audio file and resample to target sample rate. Returns mono float32."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_dBFS: float = -25.0) -> np.ndarray:
    """Normalize audio RMS to target dB FS (matches VibeVoice's AudioNormalizer)."""
    eps = 1e-6
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    if rms < eps:
        return audio
    gain = 10 ** (target_dBFS / 20.0) / rms
    return np.clip(audio * gain, -1.0, 1.0)


# ============================================================================
# Speech connector (MLP: encoder latent → LLM hidden dim)
# ============================================================================

class SpeechConnector(nn.Module):
    """Fallback MLP matching VibeVoice's connector architecture (fc1 + fc2 + norm)."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.fc2(self.act(self.fc1(x))))


# ============================================================================
# Connector loader
# ============================================================================

def load_connectors(connectors_dir: str, device: torch.device, dtype: torch.dtype):
    """Load acoustic + semantic connectors from split safetensors."""
    from safetensors.torch import load_file

    with open(os.path.join(connectors_dir, "config.json")) as f:
        config = json.load(f)

    a_dim = config["acoustic_vae_dim"]
    s_dim = config["semantic_vae_dim"]
    h_dim = config["hidden_size"]

    # Prefer the real VibeVoice class if available
    try:
        from vibevoice.modular.modeling_vibevoice import SpeechConnector as VVConn
        acoustic = VVConn(a_dim, h_dim).to(device=device, dtype=dtype)
        semantic = VVConn(s_dim, h_dim).to(device=device, dtype=dtype)
    except ImportError:
        acoustic = SpeechConnector(a_dim, h_dim).to(device=device, dtype=dtype)
        semantic = SpeechConnector(s_dim, h_dim).to(device=device, dtype=dtype)

    # Collect all safetensor shards
    state = {}
    for f in sorted(Path(connectors_dir).glob("*.safetensors")):
        state.update(load_file(str(f)))

    # Split by prefix and strip to bare names (fc1.weight, fc2.weight, etc.)
    # Handles both "acoustic_connector.fc1.weight" (new split)
    # and "model.acoustic_connector.fc1.weight" (old split)
    a_state, s_state = {}, {}
    for k, v in state.items():
        if "acoustic_connector." in k:
            bare = k.split("acoustic_connector.", 1)[1]
            a_state[bare] = v
        elif "semantic_connector." in k:
            bare = k.split("semantic_connector.", 1)[1]
            s_state[bare] = v

    # Verify weights were actually found
    if not a_state:
        raise RuntimeError(
            f"No acoustic_connector weights found in {connectors_dir}. "
            f"Keys present: {list(state.keys())[:5]}")
    if not s_state:
        raise RuntimeError(
            f"No semantic_connector weights found in {connectors_dir}. "
            f"Keys present: {list(state.keys())[:5]}")

    acoustic.load_state_dict(a_state, strict=True)
    semantic.load_state_dict(s_state, strict=True)
    acoustic.eval()
    semantic.eval()

    return acoustic, semantic, config


# ============================================================================
# Input sequence builder (matches VibeVoice-ASR training format)
# ============================================================================

SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."

def build_input_ids(
    tokenizer,
    audio_duration: float,
    vae_tok_len: int,
    speech_start_id: int,
    speech_pad_id: int,
    speech_end_id: int,
    context_info: Optional[str] = None,
) -> Tuple[List[int], List[bool]]:
    """
    Build the full input_ids matching VibeVoice-ASR's chat template.

    The sequence looks like:
      <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>
      <|im_start|>user\n<speech_start>[pad]*N<speech_end>\n{user_text}<|im_end|>
      <|im_start|>assistant\n

    Returns:
      input_ids:    List of token IDs
      acoustic_mask: List of bools — True at speech placeholder positions
    """
    show_keys = ["Start time", "End time", "Speaker ID", "Content"]

    # System turn
    sys_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}], tokenize=False)
    sys_tokens = tokenizer.encode(sys_text)

    # User turn: speech placeholders + instruction text
    sp_start = tokenizer.convert_ids_to_tokens(speech_start_id)
    sp_pad   = tokenizer.convert_ids_to_tokens(speech_pad_id)
    sp_end   = tokenizer.convert_ids_to_tokens(speech_end_id)

    if context_info and context_info.strip():
        suffix = (f"This is a {audio_duration:.2f} seconds audio, "
                  f"with extra info: {context_info.strip()}\n\n"
                  f"Please transcribe it with these keys: " + ", ".join(show_keys))
    else:
        suffix = (f"This is a {audio_duration:.2f} seconds audio, "
                  f"please transcribe it with these keys: " + ", ".join(show_keys))

    user_content = (
        "".join([sp_start] + [sp_pad] * vae_tok_len + [sp_end])
        + "\n" + suffix
    )
    user_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=True)

    # Assistant generation prefix
    gen_tokens = tokenizer.encode("<|im_start|>assistant\n")

    full = sys_tokens + user_tokens + gen_tokens
    mask = [tid == speech_pad_id for tid in full]
    return full, mask


# ============================================================================
# ExLlamaV3 generator with native MMEmbedding injection
# ============================================================================

class ExLlamaASRGenerator:
    """
    ExLlamaV3 wrapper that injects speech embeddings via the native
    MMEmbedding API — the same mechanism ExLlamaV3 uses for vision-language
    models (Gemma3, Qwen2.5-VL, Mistral3, etc.).

    How MMEmbedding works:
      1. We mark speech positions in the token string with -1
      2. MMEmbedding assigns synthetic token IDs >= 1,000,000,000
      3. During prefill, ExLlamaV3's Embedding module checks each ID:
         - ID < 1B  → normal embedding table lookup
         - ID >= 1B → fetch from our MMEmbedding.embeddings tensor
      4. The merged embeddings flow through the transformer normally

    CRITICAL: Cache must be created BEFORE model.load(). During loading,
    each attention module's load_local() calls cache_layer.alloc(device)
    to allocate the KV tensors. If the Cache doesn't exist yet, the list
    of cache layers is empty and k/v tensors are never allocated.
    """

    def __init__(self, model_dir: str, device: torch.device, cache_tokens: int = 32768):
        from exllamav3.architecture.qwen2 import Qwen2Config, Qwen2Model
        from exllamav3 import Cache, Tokenizer

        # Load config + create model skeleton
        self.config = Qwen2Config(model_dir)
        self.model = Qwen2Model(self.config)

        # Allocate KV cache BEFORE loading weights (see docstring above)
        self.cache = Cache(self.model, max_num_tokens=cache_tokens)

        # Load weights — this triggers cache_layer.alloc() inside each
        # attention module's load_local(device) method
        self.model.load()
        assert self.cache.layers[0].k is not None, "KV cache allocation failed"

        # ExLlamaV3's own tokenizer (needed by the Generator)
        self.tokenizer = Tokenizer(self.config)

        self.device = device
        self.model_dir = model_dir

        ctx = getattr(self.config, "max_seq_len", 8192)
        n_layers = len(self.cache.layers)
        print(f"  Loaded: {n_layers} layers, context {ctx}, cache {cache_tokens} tokens")

    def generate(
        self,
        speech_features: torch.Tensor,  # [1, N_speech, hidden] — encoder output
        input_ids: torch.Tensor,        # [1, seq] with speech_pad_id placeholders
        speech_pad_id: int,
        max_new_tokens: int = 8192,
        eos_token_id: int = 151643,
        stop_token_ids: Optional[List[int]] = None,
        **kwargs,                       # absorbs unused params (temperature, etc.)
    ) -> List[int]:
        """
        Generate a transcription using MMEmbedding injection.

        Steps:
          1. Find positions where input_ids == speech_pad_id
          2. Build MMEmbedding with speech features at those positions
          3. ExLlamaV3 handles the rest: paged KV cache, attention, generation
        """
        from exllamav3 import MMEmbedding
        from exllamav3.generator import Generator
        from exllamav3.generator.job import Job

        # Locate speech token positions in the input
        ids_1d = input_ids[0]
        speech_mask = (ids_1d == speech_pad_id)
        n_speech = speech_mask.sum().item()
        if n_speech == 0:
            raise ValueError(f"No speech_pad_id ({speech_pad_id}) found in input_ids")

        # Use speech features directly (already [1, N, hidden] from encoder)
        speech_emb = speech_features.squeeze(0)  # [N, hidden]

        # Build token_string: same as input_ids but -1 at speech positions.
        # MMEmbedding will replace -1s with synthetic IDs >= 1,000,000,000
        token_string = ids_1d.clone().unsqueeze(0).long().cpu()
        token_string[0, speech_mask.cpu()] = -1

        # Create MMEmbedding — embeddings on CPU to match the Embedding
        # module's prefer_cpu flag
        mme = MMEmbedding(
            embeddings=speech_emb.cpu().half(),
            token_string=token_string,
        )

        # Build stop conditions
        stop = [eos_token_id]
        if stop_token_ids:
            stop.extend(stop_token_ids)
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            try:
                tid = self.tokenizer.single_id(tok)
                if tid not in stop:
                    stop.append(tid)
            except Exception:
                pass

        # Create generator + job and run
        gen = Generator(self.model, self.cache, self.tokenizer)
        job = Job(
            input_ids=mme.token_string,    # contains synthetic IDs for speech
            max_new_tokens=max_new_tokens,
            stop_conditions=stop,
            embeddings=[mme],              # ExLlamaV3 reads embeddings from here
        )
        gen.enqueue(job)

        tokens = []
        while True:
            for r in gen.iterate():
                tids = r.get("token_ids")
                if tids is not None:
                    tokens.extend(
                        tids.flatten().tolist() if isinstance(tids, torch.Tensor)
                        else tids)
                if r.get("eos"):
                    return [t for t in tokens if t not in set(stop)]
            if not gen.pending_jobs and not getattr(gen, "active_jobs", []):
                break

        return [t for t in tokens if t not in set(stop)]


# ============================================================================
# Main pipeline
# ============================================================================

class VibeVoiceASRPipeline:
    """
    Full inference pipeline:
      audio → encoders (bf16) → connectors → embed merge → ExLlamaV3 → text

    All components load from the split directory. The original VibeVoice-ASR
    model directory is NOT required — only the split weights and the vibevoice
    Python package (for encoder class definitions).
    """

    def __init__(
        self,
        encoders_dir: str = "./split/encoders",
        connectors_dir: str = "./split/connectors",
        llm_dir: str = "./split/llm-exl3-8.0bpw",
        device: str = "cuda",
        cache_tokens: int = 32768,
    ):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        print("=" * 60)
        print("VibeVoice-ASR + ExLlamaV3 Pipeline")
        print("=" * 60)

        # 1. Tokenizer — Qwen2.5-7B vocab from HuggingFace
        print("\n[1/4] Loading tokenizer...")
        self._load_tokenizer()

        # 2. Audio encoders (bf16, from split)
        print("\n[2/4] Loading audio encoders (bf16)...")
        self._load_encoders(encoders_dir)

        # 3. Connectors (bf16, from split)
        print("\n[3/4] Loading connectors (bf16)...")
        self.acoustic_conn, self.semantic_conn, self.conn_cfg = \
            load_connectors(connectors_dir, self.device, self.dtype)
        print(f"  Acoustic: {self.conn_cfg['acoustic_vae_dim']} → {self.conn_cfg['hidden_size']}")
        print(f"  Semantic: {self.conn_cfg['semantic_vae_dim']} → {self.conn_cfg['hidden_size']}")

        # 4. LLM backbone (ExLlamaV3 quantized)
        print(f"\n[4/4] Loading ExLlamaV3 LLM...")
        self.generator = ExLlamaASRGenerator(llm_dir, self.device, cache_tokens)

        # 5. Special token IDs (Qwen2 tokens repurposed for speech by VibeVoice)
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
        self.speech_end_id   = self.tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        self.speech_pad_id   = self.tokenizer.convert_tokens_to_ids("<|box_start|>")
        self.eos_token_id    = self.tokenizer.eos_token_id
        print(f"  Speech tokens: start={self.speech_start_id}, "
              f"pad={self.speech_pad_id}, end={self.speech_end_id}")

        print(f"\n{'=' * 60}")
        print("Pipeline ready!")
        print(f"{'=' * 60}")

    def _load_tokenizer(self):
        """Load Qwen2.5-7B tokenizer with VibeVoice's chat template."""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B", trust_remote_code=True)
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    def _load_encoders(self, encoders_dir: str):
        """Load acoustic + semantic encoders from split safetensors."""
        from safetensors.torch import load_file

        try:
            from vibevoice.modular.modular_vibevoice_tokenizer import (
                VibeVoiceAcousticTokenizerModel,
                VibeVoiceSemanticTokenizerModel,
            )
            from vibevoice.modular.configuration_vibevoice import (
                VibeVoiceAcousticTokenizerConfig,
                VibeVoiceSemanticTokenizerConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                f"vibevoice package required for encoder classes: {e}\n"
                f"Install with: pip install -e /path/to/VibeVoice")

        for name, Dir, CfgCls, ModelCls, attr in [
            ("Acoustic", os.path.join(encoders_dir, "acoustic"),
             VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerModel,
             "acoustic_enc"),
            ("Semantic", os.path.join(encoders_dir, "semantic"),
             VibeVoiceSemanticTokenizerConfig, VibeVoiceSemanticTokenizerModel,
             "semantic_enc"),
        ]:
            with open(os.path.join(Dir, "config.json")) as f:
                cfg = json.load(f)
            cfg.pop("model_type", None)  # avoid HF registry errors
            model = ModelCls(CfgCls(**cfg))
            state = {}
            for sf in sorted(Path(Dir).glob("*.safetensors")):
                state.update(load_file(str(sf)))
            model.load_state_dict(state, strict=False)

            # Strip the decoder submodule — not needed for ASR, saves ~640MB VRAM
            if hasattr(model, "decoder"):
                del model.decoder
                model.decoder = None

            model.to(device=self.device, dtype=self.dtype).eval()
            setattr(self, attr, model)
            n = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {n / 1e6:.0f}M params")

        # Cache VAE sampling property
        self.std_dist_type = getattr(self.acoustic_enc, "std_dist_type", "fix")

    @torch.no_grad()
    def encode_speech(self, audio: np.ndarray, chunk_seconds: float = 30.0) -> torch.Tensor:
        sr = 24000
        chunk_samples = int(chunk_seconds * sr)
        overlap_samples = int(1.0 * sr)  # 1s overlap
        
        a_feats, s_feats = [], []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            wav = torch.from_numpy(chunk).to(device=self.device, dtype=self.dtype)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            wav3d = wav.unsqueeze(1)
            
            a_tok = self.acoustic_enc.encode(wav3d).sample(dist_type=self.std_dist_type)[0]
            a_feats.append(self.acoustic_conn(a_tok))
            s_feats.append(self.semantic_conn(self.semantic_enc.encode(wav3d).mean))
            
            # Free intermediates
            del wav, wav3d, a_tok
            torch.cuda.empty_cache()
            
            start += chunk_samples - overlap_samples
        
        # Concatenate along time dimension
        return torch.cat(a_feats, dim=1) + torch.cat(s_feats, dim=1)

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        context_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full transcription: audio file → JSON segments with timestamps + speakers.

        Returns dict with: raw_text, segments, and timing breakdown.
        """
        t0 = time.time()

        # 1. Load + normalize audio
        print(f"\n[Step 1] Loading audio: {audio_path}")
        audio = normalize_audio(load_audio(audio_path, target_sr=24000))
        duration = len(audio) / 24000.0
        print(f"  Duration: {duration:.1f}s ({len(audio):,} samples)")

        # 2. Compute expected VAE frame count (compression ratio = 3200)
        vae_frames = math.ceil(len(audio) / 3200)

        # 3. Encode speech through both encoder paths
        print(f"\n[Step 2] Encoding speech...")
        t_enc = time.time()
        speech_features = self.encode_speech(audio)
        enc_ms = (time.time() - t_enc) * 1000
        actual_frames = speech_features.shape[1]
        if actual_frames != vae_frames:
            vae_frames = actual_frames
        print(f"  Features: {speech_features.shape}, encoded in {enc_ms:.0f}ms")

        # 4. Build input token sequence (chat template with speech placeholders)
        print(f"\n[Step 3] Building input sequence...")
        ids_list, mask = build_input_ids(
            tokenizer=self.tokenizer,
            audio_duration=duration,
            vae_tok_len=vae_frames,
            speech_start_id=self.speech_start_id,
            speech_pad_id=self.speech_pad_id,
            speech_end_id=self.speech_end_id,
            context_info=context_info,
        )
        input_ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        n_speech = sum(mask)
        print(f"  Tokens: {len(ids_list)} (text: {len(ids_list) - n_speech}, speech: {n_speech})")

        # 5. Generate transcription via ExLlamaV3
        print(f"\n[Step 4] Generating (max {max_new_tokens} tokens)...")
        t_gen = time.time()
        gen_ids = self.generator.generate(
            speech_features=speech_features,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.eos_token_id,
            speech_pad_id=self.speech_pad_id,
        )
        gen_ms = (time.time() - t_gen) * 1000
        total_ms = (time.time() - t0) * 1000

        # 7. Decode tokens → text → parse JSON
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        segments = self._parse_json(raw_text)

        n_tok = len(gen_ids)
        tok_s = n_tok / (gen_ms / 1000) if gen_ms > 0 else 0
        ms_tok = gen_ms / n_tok if n_tok > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"  Audio duration:    {duration:.1f}s")
        print(f"  Encoder:           {enc_ms:.0f}ms")
        print(f"  Generation:        {gen_ms:.0f}ms  "
              f"({n_tok} tokens, {tok_s:.1f} tok/s, {ms_tok:.1f} ms/tok)")
        print(f"  Total pipeline:    {total_ms:.0f}ms")
        print(f"  Realtime factor:   {total_ms / 1000 / duration:.2f}x")
        print(f"{'=' * 60}")

        return {
            "raw_text": raw_text,
            "segments": segments,
            "audio_duration": duration,
            "num_tokens": n_tok,
            "encoder_ms": enc_ms,
            "generation_ms": gen_ms,
            "total_ms": total_ms,
            "tokens_per_second": tok_s,
            "realtime_factor": total_ms / 1000 / duration,
        }

    @staticmethod
    def _parse_json(text: str) -> List[Dict[str, Any]]:
        """Extract JSON array from model output (handles markdown code fences)."""
        try:
            if "```json" in text:
                s = text.find("```json") + 7
                e = text.find("```", s)
                return json.loads(text[s:e].strip())

            s = text.find("[")
            if s == -1:
                s = text.find("{")
            if s == -1:
                return []

            depth = 0
            for i in range(s, len(text)):
                if text[i] in "[{":
                    depth += 1
                elif text[i] in "]}":
                    depth -= 1
                    if depth == 0:
                        result = json.loads(text[s:i + 1])
                        return [result] if isinstance(result, dict) else result
        except (json.JSONDecodeError, Exception):
            pass
        return []


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="VibeVoice-ASR + ExLlamaV3 quantized inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vibevoice_asr_exl3_inference.py --audio_file recording.wav
  python vibevoice_asr_exl3_inference.py --audio_file meeting.wav --context "Alice,Bob"
  python vibevoice_asr_exl3_inference.py --audio_file call.wav --output_json out.json
        """)

    p.add_argument("--audio_file", required=True, help="Path to audio file")
    p.add_argument("--encoders_dir", default="./split/encoders")
    p.add_argument("--connectors_dir", default="./split/connectors")
    p.add_argument("--llm_dir", default="./split/llm-exl3-8.0bpw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache_tokens", type=int, default=32768,
                   help="KV cache size in tokens (default 32768)")
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--context", default=None,
                   help="Hotwords / context (comma-separated)")
    p.add_argument("--output_json", default=None, help="Save results to JSON")

    args = p.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    pipeline = VibeVoiceASRPipeline(
        encoders_dir=args.encoders_dir,
        connectors_dir=args.connectors_dir,
        llm_dir=args.llm_dir,
        device=args.device,
        cache_tokens=args.cache_tokens,
    )

    result = pipeline.transcribe(
        audio_path=args.audio_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        context_info=args.context,
    )

    # Print formatted segments
    print(f"\n{'=' * 60}")
    print("TRANSCRIPTION")
    print(f"{'=' * 60}\n")

    if result["segments"]:
        for seg in result["segments"]:
            start   = seg.get("Start time", seg.get("Start", seg.get("start", "?")))
            end     = seg.get("End time",   seg.get("End",   seg.get("end",   "?")))
            speaker = seg.get("Speaker ID", seg.get("Speaker", seg.get("speaker", "?")))
            content = seg.get("Content",    seg.get("content", seg.get("text", "")))
            print(f"  [{start} -> {end}] Speaker {speaker}: {content}")
    else:
        print(result["raw_text"])

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {args.output_json}")


if __name__ == "__main__":
    main()
