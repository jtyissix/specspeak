import argparse
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute WER using Whisper-large-v3.
Inputs: audio_paths (list[str]) and references (list[str]) in the script, or load them yourself.
Outputs: total WER + per-item WER, and optionally a JSONL with details.

Install:
  pip install -U transformers accelerate jiwer torch soundfile
Optional (if soundfile can't read your audio type or you need resampling):
  pip install -U librosa
"""

import os
import re
import json
from typing import List, Tuple, Optional

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


from qwen_asr import Qwen3ASRModel
from jiwer import (
    Compose,
    RemovePunctuation,
    ToLowerCase,
    RemoveMultipleSpaces,
    Strip,
    #ReplaceRegex,
    wer,
)

# Audio backends
try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

import re
import unicodedata

class KeepEnglishAndDigits:
    def __call__(self, s):
        # 只保留英文字母、数字、空格
        return re.sub(r'[^a-zA-Z0-9\s]', '', s)
class KeepEnglishOnly:
    def __call__(self, s):
        # 只保留英文字母和空格
        return re.sub(r'[^a-zA-Z\s]', '', s)
# ----------------------------
# Text normalization (both sides)
# ----------------------------
JIWER_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    KeepEnglishOnly(),
    Strip(),
])
def remove_audio_tokens(text):
    pattern = re.compile(r"<\|audio_\d+\|>|<\|begin_of_audio\|>|<\|end_of_audio\|>|<\|im_end\|>")
    return pattern.sub("", text)
def normalize_text(s: str) -> str:
    # jiwer pipeline; keep as a wrapper for clarity
    return JIWER_TRANSFORM(s)


# ----------------------------
# Audio loading
# ----------------------------
def _resample_np(x, orig_sr: int, target_sr: int):
    if not HAVE_LIBROSA:
        raise RuntimeError("Need librosa for resampling. Install: pip install librosa")
    return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)

def load_audio_16k_mono(path: str) -> torch.Tensor:
    """
    Returns waveform as torch.float32 with shape [T] at 16kHz mono.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    if HAVE_SF:
        wav, sr = sf.read(path, always_2d=False)
        # multi-channel -> mono
        if hasattr(wav, "ndim") and wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav = wav.astype("float32", copy=False)

        if sr != 16000:
            wav = _resample_np(wav, orig_sr=sr, target_sr=16000).astype("float32", copy=False)

        return torch.from_numpy(wav)

    if HAVE_LIBROSA:
        wav, _ = librosa.load(path, sr=16000, mono=True)
        return torch.tensor(wav, dtype=torch.float32)

    raise RuntimeError("No audio backend available. Install soundfile or librosa.")


# ----------------------------
# Whisper transcription
# ----------------------------
@torch.inference_mode()
def transcribe_whisper_large_v3(
    audio_paths: List[str],
    batch_size: int = 4,
    language: Optional[str] = None,   # e.g. "en"; None = auto
    device: Optional[str] = None,
) -> List[str]:
    """
    Returns hypothesis texts in the same order as audio_paths.
    """
    model_id = "Qwen/Qwen3-ASR-1.7B"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=16, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )
    results = model.transcribe(
    audio=audio_paths,
    language="English", # set "English" to force the language
    )
    return [result.text for result in results]
    '''
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    hyps: List[str] = []

    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start:start + batch_size]

        waves = []
        for p in batch_paths:
            wav = load_audio_16k_mono(p)
            waves.append(wav.numpy())

        inputs = processor(
            waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_features = inputs.input_features.to(device, dtype=dtype)

        gen_kwargs = {}
        # transformers Whisper supports forced language via generate kwargs on many versions;
        # if your version errors, set language=None.
        if language is not None:
            gen_kwargs["language"] = language

        pred_ids = model.generate(input_features, **gen_kwargs)
        texts = processor.batch_decode(pred_ids, skip_special_tokens=True)

        hyps.extend(texts)

    return hyps
    '''

# ----------------------------
# WER computation
# ----------------------------
def compute_wer_with_whisper_large_v3(
    audio_paths: List[str],
    references: List[str],
    batch_size: int = 4,
    language: Optional[str] = None,
    save_jsonl: Optional[str] = None,
) -> Tuple[float, List[float], List[str]]:
    """
    Returns:
      total_wer, per_item_wer_list, hypothesis_text_list
    """
    if len(audio_paths) != len(references):
        raise ValueError(f"Length mismatch: audio_paths={len(audio_paths)} refs={len(references)}")

    hyps = transcribe_whisper_large_v3(
        audio_paths=audio_paths,
        batch_size=batch_size,
        language=language,
    )
    #breakpoint()
    refs_norm = [normalize_text(remove_audio_tokens(r)) for r in references]
    hyps_norm = [normalize_text(remove_audio_tokens(h)) for h in hyps]

    total = wer(refs_norm, hyps_norm)
    per_item = [wer([r], [h]) for r, h in zip(refs_norm, hyps_norm)]

    if save_jsonl:
        os.makedirs(os.path.dirname(save_jsonl), exist_ok=True)
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for i, (ap, r_raw, h_raw, r_n, h_n, w) in enumerate(
                zip(audio_paths, references, hyps, refs_norm, hyps_norm, per_item)
            ):
                f.write(json.dumps({
                    "index": i,
                    "audio_path": ap,
                    "ref_raw": r_raw,
                    "hyp_raw": h_raw,
                    "ref_norm": r_n,
                    "hyp_norm": h_n,
                    "wer": w,
                }, ensure_ascii=False) + "\n")

    return total, per_item, hyps


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default="output/ablation/webq/no_two_vo/generated_text.jsonl")
    parser.add_argument("--save_jsonl", default="output/whisper_large_v3_wer_detail.jsonl")
    args = parser.parse_args()
    # ====== Replace with your real lists ======
    '''
    audio_paths = [
        "asset/0.wav",
        "asset/1.wav",
    ]
    references = [
        "Hello, World! This is a TEST.",
        "Another sentence; with Punctuation & Caps.",
    ]
    '''
    # ========================================
    data = []
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            #breakpoint()
            #print(line)
            data.append(json.loads(line))
    #audio_paths = [item["audio_output"] for item in data]
    #references = [item["text"] for item in data]
    audio_paths = [item["audio_output"] for i, item in enumerate(data) if item["text"]!="" and item["text"]!="<|im_end|>"]
    references  = [item["text"]         for i, item in enumerate(data) if item["text"]!="" and item["text"]!="<|im_end|>"]
    total_wer, per_item_wer, hyps = compute_wer_with_whisper_large_v3(
        audio_paths=audio_paths,
        references=references,
        batch_size=16,
        language="en",  # set None if you want auto language detection
        save_jsonl=args.save_jsonl,
    )

    print("Total WER:", total_wer)

    # Show worst 10
    worst = sorted(enumerate(per_item_wer), key=lambda x: x[1], reverse=True)[:20]
    print("\nWorst samples (idx, wer, path):")
    for idx, w in worst:
        print(idx, w, audio_paths[idx])