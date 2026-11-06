# Speak Early: Speculative Speech Generation for Low-Latency SpeechLLMs



**SpecSpeak** reduces Time-to-First-Speech (TTFS) by overlapping language model decoding with diffusion-based speech synthesis. A lightweight draft model produces an initial prefix and triggers speech synthesis while the target model verifies the draft. Verification-aware reuse preserves intermediate diffusion states and incorporates corrected tokens to deliver the first speech chunk earlier.

This repository provides the implementation of SpecSpeak.

## 🔥 News

- **2026.09** — 🎉 Our paper is accepted to **NeurIPS 2026**!
- **2026.11** — Code has been released.

## :label: TODO 

- [x] Release implementation of VITA-Audio
- [x] Release draft model and TRT Decoder
- [ ] Release implementation of VocalNet

## 📄 Contents


- [Installation](#-installation)
- [Prepare Checkpoints](#-prepare-checkpoints)
- [Quick Start](#-quick-start)
- [Build the TensorRT Decoder](#-build-the-tensorrt-decoder)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)



## 📔 Installation



### Get the code and start the container

```bash
git clone https://github.com/jtyissix/specspeak.git
cd specspeak
```

Run the following commands inside the container, from the directory containing `setup.py`:

```bash
pip install -r requirements.txt
pip install -r requirements_ds_gpu.txt
pip install -e .
pip install loguru soundfile onnx "tensorrt-cu12>=10,<11"
pip install flash-attn --no-build-isolation

export PYTHONPATH="$PWD:$PWD/third_party/GLM-4-Voice:$PWD/third_party/GLM-4-Voice/cosyvoice:$PWD/third_party/GLM-4-Voice/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
```

Run all commands below from this same directory and shell.

## 📦 Prepare Checkpoints

Download the backbone, speech tokenizer, and speech decoder:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "VITA-MLLM/VITA-Audio-Balance",
    local_dir="models/vita_balance_official",
)
snapshot_download(
    "zai-org/glm-4-voice-tokenizer",
    local_dir="models/THUDM",
)
snapshot_download(
    "zai-org/glm-4-voice-decoder",
    local_dir="Decoder",
)
PY
```

- **7B backbone:** [VITA-Audio-Balance](https://huggingface.co/VITA-MLLM/VITA-Audio-Balance).
- **0.5B draft checkpoint:** [Download](https://huggingface.co/jtyissix/VITA-Audio-Draft-0.5B). Extract to `models/vita_0.5b_balance_final/`.
- **Speech tokenizer:** [GLM-4-Voice-Tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer).
- **Speech decoder:** [GLM-4-Voice-Decoder](https://huggingface.co/zai-org/glm-4-voice-decoder).
- **TensorRT decoder:** [Download](https://huggingface.co/jtyissix/glm-4-voice-decoder-trt). Place the engine at `models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan`.

<!-- Replace DRAFT_MODEL_URL and TRT_DECODER_URL with the release links. -->

Arrange the files as follows:

```text
.
├── models/
│   ├── vita_balance_official/
│   ├── vita_0.5b_balance_final/
│   ├── THUDM/
│   └── Decoder_trt/
│       └── flow.decoder.estimator.fp32.a800.plan
├── Decoder/
│   ├── config.yaml
│   ├── flow.pt
│   └── hift.pt
├── asset/
├── infer/
└── output/
```

To build an engine on your GPU, follow [Build the TensorRT Decoder](#-build-the-tensorrt-decoder).

## 🚀 Quick Start

### 1. Configure checkpoint paths

Open `infer/vitaspecdiff_alp.py` and set the checkpoint paths at the top of the file. Set the TensorRT engine path in `VitaStreaming.__init__`.

| Setting | Location | Value |
| :--- | :--- | :--- |
| Target checkpoint | `target_model_name_or_path` | `models/vita_balance_official/` |
| Draft checkpoint | `draft_model_name_or_path` | `models/vita_0.5b_balance_final/` |
| Audio tokenizer | `audio_tokenizer_path` | `models/THUDM/` |
| Decoder checkpoint | `flow_path` | `Decoder/` |
| TensorRT engine | `trt_path` in `VitaStreaming.__init__` | `models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan` |
| GPU | `CUDA_VISIBLE_DEVICES` | `0` |

The implementation uses BF16 and FlashAttention 2 for both LLMs. The first prefix contains two text tokens; diffusion runs for ten steps, with reuse limits of four and two steps for the correction branches.

### 2. Configure inputs and outputs

In the `if __name__ == "__main__":` block, set the audio directory, input CSV, and output paths:

```python
audio_folder = "datasets/alpaca_eval/audios"
data_csv_path = "datasets/alpaca_eval/alpaca_eval.csv"
audio_save_path = "output/specspeak/audio"
text_jsonl_path = "output/specspeak/responses.jsonl"
time_json_path = "output/specspeak/latency.json"
```

The CSV has a header row and stores the audio filename in its **fifth column** (`row[4]`). Each filename is resolved relative to `audio_folder`; adjust `row[4]` if your CSV uses a different column.

Before `vita = VitaStreaming()`, create the output directories and initialize the latency records:

```python
os.makedirs(audio_save_path, exist_ok=True)
os.makedirs(os.path.dirname(text_jsonl_path), exist_ok=True)
os.makedirs(os.path.dirname(time_json_path), exist_ok=True)
total_time_list = []
num_dict = {}
```

### 3. Run accelerated inference

From the repository root, run:

```bash
CUDA_VISIBLE_DEVICES=0 python infer/vitaspecdiff_alp.py
```

The script processes the recordings listed in the CSV and saves:

- **Generated speech** to `audio_save_path`, at **22,050 Hz**.
- **Response text and audio paths** to `text_jsonl_path`, one JSON record per input.
- **TTFS values in seconds** to `time_json_path`, under the `our_time` key.

The terminal prints each response's first-audio latency, followed by average, fastest, and P90 latency across the inputs.

## ⚡ Build the TensorRT Decoder

Export the diffusion estimator to ONNX, then build its FP32 TensorRT engine on the inference GPU:

```bash
mkdir -p models/Decoder_trt

python scripts/pt2onnx.py \
  --config_path Decoder/config.yaml \
  --flow_ckpt_path Decoder/flow.pt \
  --save_path models/Decoder_trt/flow.decoder.estimator.fp32.onnx

python scripts/onnx2trt.py \
  --onnx_path models/Decoder_trt/flow.decoder.estimator.fp32.onnx \
  --engine_path models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan
```

The export script checks ONNX outputs against PyTorch. The engine builder then loads the serialized engine and prints its input/output tensor information. The generated engine path matches the SpecSpeak quick-start configuration.

## 🖋️ Citation

```bibtex
@inproceedings{jiang2026speakearly,
  title     = {Speak Early: Speculative Speech Generation for Low-Latency SpeechLLMs},
  author    = {Jiang, Tianyuan and Deng, Yongheng and Ren, Ju},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This implementation builds on [VITA-Audio](https://github.com/VITA-MLLM/VITA-Audio), [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice), [VocalNet](https://github.com/SJTU-OmniAgent/VocalNet) and [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS). We thank their authors for sharing the code and checkpoints.
