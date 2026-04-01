#!/usr/bin/env python3
"""
VibeVoice-ASR Weight Splitter
==============================
Splits Microsoft's VibeVoice-ASR (~17GB) into quantization-ready components:

  split/
  ├── llm/              Qwen2-7B backbone (14.2GB bf16 → quantize to ~7GB)
  ├── encoders/
  │   ├── acoustic/     Acoustic encoder + config (~1GB, keep bf16)
  │   └── semantic/     Semantic encoder + config (~0.9GB, keep bf16)
  └── connectors/       Projection MLPs + config (~100MB, keep bf16)

After splitting, quantize the LLM with ExLlamaV3:
  python convert.py -i ./split/llm -o ./split/llm-exl3-8.0bpw -b 8.0

Then run inference with vibevoice_asr_exl3_inference.py (~10GB VRAM, 4.7x faster).

Architecture (from VibeVoice-ASR's config):
  model.language_model.*           → Qwen2-7B (28 layers, 7.6B params)
  model.acoustic_tokenizer.*       → Acoustic VAE encoder + decoder
  model.semantic_tokenizer.*       → Semantic encoder
  model.acoustic_connector.*       → MLP: acoustic latent → LLM hidden (3584)
  model.semantic_connector.*       → MLP: semantic latent → LLM hidden (3584)
  lm_head.weight                   → Output projection

Only the encoder halves of the tokenizers are needed for ASR inference.
The acoustic decoder (~640MB) is skipped by default.

Usage:
  # Preview what will be split (no files written):
  python split_vibevoice_asr.py --model_dir ./VibeVoice-ASR --dry_run

  # Split weights:
  python split_vibevoice_asr.py --model_dir ./VibeVoice-ASR

Requirements:
  pip install torch safetensors
"""

import argparse
import json
import os
import struct
import sys
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import torch
    from safetensors.torch import load_file, save_file
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# ============================================================================
# Weight classification
# ============================================================================

def classify(name: str) -> str:
    """Classify a weight tensor name into its component group."""
    if name.startswith("model.acoustic_connector."):  return "connector"
    if name.startswith("model.semantic_connector."):   return "connector"
    if name.startswith("model.acoustic_tokenizer.encoder."): return "acoustic_encoder"
    if name.startswith("model.acoustic_tokenizer.decoder."): return "acoustic_decoder"
    if name.startswith("model.acoustic_tokenizer."):   return "acoustic_encoder"
    if name.startswith("model.semantic_tokenizer."):    return "semantic_encoder"
    if name.startswith("model.language_model."):        return "llm"
    if name == "lm_head.weight":                        return "llm"
    return "unknown"


def rename_for_qwen(name: str) -> str:
    """
    Rename LLM weights from VibeVoice's namespace to standard Qwen2 format.
    model.language_model.layers.0.* → model.layers.0.*
    model.language_model.embed_tokens.* → model.embed_tokens.*
    lm_head.weight stays unchanged.
    """
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    return name


# ============================================================================
# Utilities
# ============================================================================

def bytes_to_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def peek_headers(model_dir: Path) -> dict:
    """Read safetensors headers without loading tensor data (fast, ~0 RAM)."""
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No .safetensors files in {model_dir}")
        sys.exit(1)

    tensors = {}
    for st_file in st_files:
        with open(st_file, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            shape = meta.get("shape", [])
            offsets = meta.get("data_offsets", [0, 0])
            n_params = 1
            for s in shape:
                n_params *= s
            tensors[key] = {
                "shape": shape,
                "dtype": meta.get("dtype", "BF16"),
                "file": st_file.name,
                "num_params": n_params,
                "size_bytes": offsets[1] - offsets[0] if len(offsets) == 2 else n_params * 2,
            }
    return tensors


def build_qwen2_config(original_config: dict) -> dict:
    """Extract a standalone Qwen2ForCausalLM config from VibeVoice's nested config."""
    d = original_config.get("decoder_config", {})
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "attention_dropout": d.get("attention_dropout", 0.0),
        "bos_token_id": original_config.get("bos_token_id", 151643),
        "eos_token_id": original_config.get("eos_token_id", 151645),
        "hidden_act": d.get("hidden_act", "silu"),
        "hidden_size": d.get("hidden_size", 3584),
        "initializer_range": d.get("initializer_range", 0.02),
        "intermediate_size": d.get("intermediate_size", 18944),
        "max_position_embeddings": d.get("max_position_embeddings", 131072),
        "max_window_layers": d.get("max_window_layers", 28),
        "num_attention_heads": d.get("num_attention_heads", 28),
        "num_hidden_layers": d.get("num_hidden_layers", 28),
        "num_key_value_heads": d.get("num_key_value_heads", 4),
        "rms_norm_eps": d.get("rms_norm_eps", 1e-6),
        "rope_scaling": d.get("rope_scaling", None),
        "rope_theta": d.get("rope_theta", 1000000.0),
        "sliding_window": d.get("sliding_window", None),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_mrope": d.get("use_mrope", False),
        "use_sliding_window": d.get("use_sliding_window", False),
        "vocab_size": d.get("vocab_size", 152064),
    }


def extract_encoder_configs(original_config: dict, connectors_dir: Path,
                            acoustic_dir: Path, semantic_dir: Path,
                            conn_weights: dict = None):
    """
    Write standalone config.json for encoders and connectors, extracted
    from VibeVoice-ASR's nested config. These are needed at inference time
    to instantiate the encoder classes without the original model directory.
    """
    decoder_cfg = original_config.get("decoder_config", original_config)
    hidden_size = decoder_cfg.get("hidden_size", 3584)

    # Infer connector input dims from actual weight shapes if available
    acoustic_vae_dim = 64
    semantic_vae_dim = 64
    if conn_weights:
        for k, v in conn_weights.items():
            if "acoustic_connector.fc1.weight" in k:
                acoustic_vae_dim = v.shape[1]
            elif "semantic_connector.fc1.weight" in k:
                semantic_vae_dim = v.shape[1]

    # Connector config (dimensions for the projection MLPs)
    with open(connectors_dir / "config.json", "w") as f:
        json.dump({
            "acoustic_vae_dim": int(acoustic_vae_dim),
            "semantic_vae_dim": int(semantic_vae_dim),
            "hidden_size": hidden_size,
        }, f, indent=2)

    # Find encoder sub-configs in the top-level config
    acoustic_cfg, semantic_cfg = {}, {}
    for k, v in original_config.items():
        if isinstance(v, dict):
            if "acoustic" in k.lower():
                acoustic_cfg = v
            if "semantic" in k.lower():
                semantic_cfg = v

    if not acoustic_cfg:
        acoustic_cfg = {"model_type": "vibevoice_acoustic"}
    if not semantic_cfg:
        semantic_cfg = {"model_type": "vibevoice_semantic"}

    with open(acoustic_dir / "config.json", "w") as f:
        json.dump(acoustic_cfg, f, indent=2)
    with open(semantic_dir / "config.json", "w") as f:
        json.dump(semantic_cfg, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Split VibeVoice-ASR into quantization-ready components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After splitting:
  1. Quantize LLM:  python convert.py -i ./split/llm -o ./split/llm-exl3-8.0bpw -b 8.0
  2. Inference:      python vibevoice_asr_exl3_inference.py --audio_file recording.wav
        """)

    p.add_argument("--model_dir", required=True, help="Path to VibeVoice-ASR model")
    p.add_argument("--output_dir", default="./split", help="Output directory")
    p.add_argument("--keep_acoustic_decoder", action="store_true",
                   help="Include acoustic decoder (~640MB, not needed for ASR)")
    p.add_argument("--shard_size_gb", type=float, default=2.0,
                   help="Max shard size in GB for LLM output")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview classification without writing files")

    args = p.parse_args()

    if not args.dry_run and not HAS_DEPS:
        print("ERROR: Need torch + safetensors.")
        print("  pip install torch safetensors")
        print("  (Use --dry_run to preview without them)")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    # -- Read config --
    config_path = model_dir / "config.json"
    original_config = {}
    if config_path.exists():
        with open(config_path) as f:
            original_config = json.load(f)
        print(f"Model: {original_config.get('model_type', '?')}")
        print(f"Arch:  {original_config.get('architectures', '?')}\n")

    # =====================================================================
    # Phase 1: Classify all weights (header-only scan, ~0 RAM)
    # =====================================================================
    print("Scanning weight headers...")
    all_tensors = peek_headers(model_dir)
    print(f"Found {len(all_tensors)} tensors\n")

    groups = defaultdict(list)
    group_params = defaultdict(int)
    group_bytes = defaultdict(int)

    for name, info in sorted(all_tensors.items()):
        cat = classify(name)
        groups[cat].append(name)
        group_params[cat] += info["num_params"]
        group_bytes[cat] += info["size_bytes"]

    # Print summary table
    print(f"{'Component':<22} {'Tensors':>8} {'Params':>14} {'Size':>10}")
    print("-" * 60)
    for cat in ["llm", "connector", "acoustic_encoder", "acoustic_decoder",
                "semantic_encoder", "unknown"]:
        if cat not in groups:
            continue
        note = ""
        if cat == "acoustic_decoder" and not args.keep_acoustic_decoder:
            note = " (skipped)"
        print(f"  {cat:<20} {len(groups[cat]):>6}   "
              f"{group_params[cat]/1e9:>10.3f}B   "
              f"{bytes_to_human(group_bytes[cat]):>10}{note}")
    print()

    # VRAM estimates at common quantization levels
    llm_params = group_params.get("llm", 0)
    enc_bytes = (group_bytes.get("connector", 0) +
                 group_bytes.get("acoustic_encoder", 0) +
                 group_bytes.get("semantic_encoder", 0))

    print("Quantization estimates (LLM + encoders/connectors):")
    for bpw in [8.0, 6.0, 4.0]:
        q = llm_params * bpw / 8
        print(f"  {bpw} bpw: {bytes_to_human(q)} + {bytes_to_human(enc_bytes)} "
              f"= {bytes_to_human(q + enc_bytes)}")
    print()

    if groups.get("unknown"):
        print("WARNING: unclassified weights:")
        for name in groups["unknown"]:
            print(f"  {name}  {all_tensors[name]['shape']}")
        print()

    if args.dry_run:
        print("[DRY RUN] Would write to:")
        print(f"  {output_dir}/llm/              {len(groups.get('llm', []))} tensors")
        print(f"  {output_dir}/encoders/acoustic/ {len(groups.get('acoustic_encoder', []))} tensors")
        print(f"  {output_dir}/encoders/semantic/ {len(groups.get('semantic_encoder', []))} tensors")
        print(f"  {output_dir}/connectors/        {len(groups.get('connector', []))} tensors")
        return

    # =====================================================================
    # Phase 2: Load all weights into RAM and sort by component
    # =====================================================================
    print("=" * 60)
    print("Loading weights (~17GB RAM required)...")
    print("=" * 60)

    state_dict = {}
    for st_file in sorted(model_dir.glob("*.safetensors")):
        print(f"  {st_file.name}...")
        state_dict.update(load_file(str(st_file), device="cpu"))
    print(f"  Loaded {len(state_dict)} tensors\n")

    llm_w, conn_w, a_enc_w, s_enc_w, dec_w, other_w = {}, {}, {}, {}, {}, {}

    for name, tensor in state_dict.items():
        cat = classify(name)
        if cat == "llm":
            llm_w[rename_for_qwen(name)] = tensor
        elif cat == "connector":
            # Strip "model." prefix so keys become "acoustic_connector.fc1.weight" etc.
            conn_w[name.replace("model.", "", 1)] = tensor
        elif cat == "acoustic_encoder":
            a_enc_w[name.replace("model.acoustic_tokenizer.", "")] = tensor
        elif cat == "semantic_encoder":
            s_enc_w[name.replace("model.semantic_tokenizer.", "")] = tensor
        elif cat == "acoustic_decoder":
            dec_w[name] = tensor
        else:
            other_w[name] = tensor

    del state_dict

    # =====================================================================
    # Phase 3: Write each component to disk
    # =====================================================================

    # -- LLM backbone (sharded safetensors + Qwen2 config + tokenizer) --
    llm_dir = output_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LLM ({len(llm_w)} tensors)...")

    shard_max = int(args.shard_size_gb * 1024 ** 3)
    shard, shard_bytes, shard_keys = {}, 0, []
    shard_idx = 1
    shard_names = []

    for name in sorted(llm_w.keys()):
        t = llm_w[name]
        tb = t.numel() * t.element_size()
        if shard_bytes + tb > shard_max and shard:
            fn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file(shard, str(llm_dir / fn))
            shard_names.append((shard_idx, fn, list(shard_keys)))
            print(f"  Shard {shard_idx}: {bytes_to_human(shard_bytes)}")
            shard_idx += 1
            shard, shard_bytes, shard_keys = {}, 0, []
        shard[name] = t
        shard_bytes += tb
        shard_keys.append(name)

    if shard:
        fn = "model.safetensors" if shard_idx == 1 else \
            f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(shard, str(llm_dir / fn))
        shard_names.append((shard_idx, fn, list(shard_keys)))
        print(f"  Shard {shard_idx}: {bytes_to_human(shard_bytes)}")

    total_shards = shard_idx
    weight_map = {}
    if total_shards > 1:
        for idx, old_fn, keys in shard_names:
            new_fn = old_fn.replace("PLACEHOLDER", f"{total_shards:05d}")
            old_p, new_p = llm_dir / old_fn, llm_dir / new_fn
            if old_p.exists() and old_fn != new_fn:
                old_p.rename(new_p)
            for k in keys:
                weight_map[k] = new_fn
        with open(llm_dir / "model.safetensors.index.json", "w") as f:
            json.dump({
                "metadata": {"total_size": sum(
                    t.numel() * t.element_size() for t in llm_w.values())},
                "weight_map": dict(sorted(weight_map.items())),
            }, f, indent=2)
    else:
        for _, fn, keys in shard_names:
            for k in keys:
                weight_map[k] = fn

    # Write Qwen2 config (verify vocab size against actual embedding)
    qwen_cfg = build_qwen2_config(original_config)
    for name, t in llm_w.items():
        if "embed_tokens" in name:
            if t.shape[0] != qwen_cfg["vocab_size"]:
                qwen_cfg["vocab_size"] = t.shape[0]
            break
    with open(llm_dir / "config.json", "w") as f:
        json.dump(qwen_cfg, f, indent=2)

    # Copy tokenizer files from original model
    for tf in ["tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.json",
               "merges.txt", "added_tokens.json"]:
        src = model_dir / tf
        if src.exists():
            shutil.copy2(src, llm_dir / tf)

    with open(llm_dir / "generation_config.json", "w") as f:
        json.dump({
            "bos_token_id": qwen_cfg.get("bos_token_id", 151643),
            "eos_token_id": qwen_cfg.get("eos_token_id", 151645),
            "do_sample": True, "temperature": 0.6, "top_p": 0.9,
        }, f, indent=2)

    n = sum(t.numel() for t in llm_w.values())
    b = sum(t.numel() * t.element_size() for t in llm_w.values())
    print(f"  Total: {n / 1e9:.2f}B params, {bytes_to_human(b)}")
    del llm_w

    # -- Encoders + connectors --
    a_dir = output_dir / "encoders" / "acoustic"
    s_dir = output_dir / "encoders" / "semantic"
    c_dir = output_dir / "connectors"
    for d in [a_dir, s_dir, c_dir]:
        d.mkdir(parents=True, exist_ok=True)

    extract_encoder_configs(original_config, c_dir, a_dir, s_dir,
                            conn_weights=conn_w)

    for label, weights, out_dir in [
        ("Acoustic encoder", a_enc_w, a_dir),
        ("Semantic encoder", s_enc_w, s_dir),
        ("Connectors", conn_w, c_dir),
    ]:
        if weights:
            fname = "connectors.safetensors" if "Connector" in label else "model.safetensors"
            save_file(weights, str(out_dir / fname))
            n = sum(t.numel() for t in weights.values())
            b = sum(t.numel() * t.element_size() for t in weights.values())
            print(f"  {label}: {len(weights)} tensors, "
                  f"{n / 1e6:.0f}M params, {bytes_to_human(b)}")
    del a_enc_w, s_enc_w, conn_w

    # -- Acoustic decoder (optional) --
    if args.keep_acoustic_decoder and dec_w:
        dec_dir = output_dir / "acoustic_decoder"
        dec_dir.mkdir(parents=True, exist_ok=True)
        save_file(dec_w, str(dec_dir / "acoustic_decoder.safetensors"))
        n = sum(t.numel() for t in dec_w.values())
        print(f"  Acoustic decoder: {len(dec_w)} tensors, {n / 1e9:.3f}B params")
    elif dec_w:
        b = sum(t.numel() * t.element_size() for t in dec_w.values())
        print(f"  Acoustic decoder: SKIPPED ({bytes_to_human(b)}, not needed for ASR)")
    del dec_w

    if other_w:
        od = output_dir / "other"
        od.mkdir(parents=True, exist_ok=True)
        save_file(other_w, str(od / "other.safetensors"))
        print(f"  Unclassified: {len(other_w)} tensors")
    del other_w

    # Save original config for reference
    with open(output_dir / "original_config.json", "w") as f:
        json.dump(original_config, f, indent=2)

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("SPLIT COMPLETE")
    print(f"{'=' * 60}")
    print(f"""
Output: {output_dir}/
  llm/                 Qwen2-7B backbone (ready for ExLlamaV3 quantization)
  encoders/acoustic/   Acoustic encoder + config (bf16)
  encoders/semantic/   Semantic encoder + config (bf16)
  connectors/          Projection MLPs + config (bf16)

Next steps:

  1. Quantize the LLM with ExLlamaV3:
     python convert.py -i {llm_dir} -o {output_dir}/llm-exl3-8.0bpw -b 8.0

  2. Run inference (~10GB VRAM, ~55 tok/s):
     python vibevoice_asr_exl3_inference.py --audio_file recording.wav

  3. You can now delete the original 17GB model directory.
""")


if __name__ == "__main__":
    main()
