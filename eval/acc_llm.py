import argparse
import json
import os
import re
import time
from typing import List, Dict, Any, Optional
import csv
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prompt(question: str, gold_answers: List[str], model_answer: str) -> str:
    """
    让模型输出严格 JSON: {"verdict":"YES"} 或 {"verdict":"NO"}
    """
    gold_block = "\n".join([f"- {a}" for a in gold_answers])
    prompt = (
        "You are a strict and fair judge for QA evaluation.\n"
        "Given a QUESTION, a list of GOLD ANSWERS (aliases allowed), and a MODEL ANSWER,\n"
        "decide whether the model answer should be counted as correct.\n\n"
        "Rules:\n"
        "1) Accept if the model answer clearly provides the same factual answer as ANY gold answer, even if phrased differently.\n"
        "2) Reject if it is ambiguous, contradicts, or gives a different entity/value.\n"
        "3) If the model answer contains extra text but includes the correct answer unambiguously, accept.\n"
        "4) If the question expects a short entity/value, do NOT accept long unrelated passages unless the answer is explicit.\n"
        "5) Output ONLY valid JSON with key verdict. No other text.\n"
        "Return exactly: {\"verdict\":\"YES\"} or {\"verdict\":\"NO\"}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWERS:\n{gold_block}\n\n"
        f"MODEL ANSWER:\n{model_answer}\n\n"
        "Now judge correctness. Output ONLY JSON."
    )
    return prompt


def parse_verdict(text: str) -> Optional[bool]:
    """
    解析 {"verdict":"YES"} / {"verdict":"NO"}
    返回 True/False/None
    """
    if text is None:
        return None
    t = text.strip()

    # 去掉可能的 ```json ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()

    # 尝试抽取 JSON 子串（模型偶尔会多吐字）
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0).strip()

    try:
        obj = json.loads(t)
        v = str(obj.get("verdict", "")).strip().upper()
        if v == "YES":
            return True
        if v == "NO":
            return False
        return None
    except Exception:
        return None


@torch.inference_mode()
def batch_judge(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    max_new_tokens: int = 32,
) -> List[str]:
    """
    对一批 prompts 生成裁决 JSON。
    """
    # Qwen 系列一般走 chat template 更稳
    # 这里把 prompt 放 user role，system 规则已经写进 prompt 文本里了
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]

    # apply_chat_template 会加上必要的特殊 token
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 只取新生成部分
    gen = outputs[:, inputs.shape[1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto", help='e.g. "auto" or "cuda:0"')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1, help="debug: limit samples, -1 = all")
    parser.add_argument("--sleep", type=float, default=0.0, help="optional sleep between batches")
    parser.add_argument("--pred_path", default="output/ablation/webq/no_two_vo/generated_text.jsonl")
    parser.add_argument("--data_csv_path", default="datasets/web_questions/eval_datas/web_questions/web_questions.csv")
    args = parser.parse_args()

    pred_path = args.pred_path
    data_csv_path = args.data_csv_path
    
    data: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    has_csv=True
    print("Loading dataset...")
    
    if has_csv:
        dataset=[]
        with open(data_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            header = next(reader)  # 跳过表头

            for row in reader:
                question= row[0]  # 假设第5列是 wav 文件名
                answer=row[1]
                dataset.append({"question":question,"answer":answer})
    else:
        dataset = load_dataset("fixie-ai/llama-questions", split="test")
        dataset = dataset.remove_columns(["audio"])

    if len(data) != len(dataset):
        raise ValueError(f"Prediction length {len(data)} != dataset length {len(dataset)}")

    # ---- load model/tokenizer (offline) ----
    # 你想完全离线：确保本地 HF cache 已经有模型；或者提前 `HF_HUB_OFFLINE=1`
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading model: {args.model_name} (dtype={args.dtype}, device_map={args.device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    total = len(dataset) if args.max_samples < 0 else min(len(dataset), args.max_samples)

    correct = 0
    judged = 0

    # batch 推理
    bs = max(1, args.batch_size)

    pbar = tqdm(range(0, total, bs), desc="LLM judging (offline)")
    for start in pbar:
        end = min(start + bs, total)

        prompts = []
        batch_meta = []
        for i in range(start, end):
            pred_text = str(data[i]["text"]).strip()
            idx = data[i]["index"]
            item = dataset[idx]
            question = item.get("question", "")
            gt_answers = item.get("answer", [])
            if isinstance(gt_answers, str):
                gt_answers = [gt_answers]
            gt_answers = [str(x) for x in gt_answers]

            prompts.append(build_prompt(question, gt_answers, pred_text))
            batch_meta.append((i, idx))

        gen_texts = batch_judge(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )

        for out in gen_texts:
            verdict = parse_verdict(out)
            hit = bool(verdict) if verdict is not None else False
            judged += 1
            if hit:
                correct += 1

        pbar.set_postfix(acc=f"{(correct/judged):.4f}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    accuracy = correct / judged if judged else 0.0

    print("=" * 50)
    print(f"Total samples judged: {judged}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()