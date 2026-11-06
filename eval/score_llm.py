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
    AlpacaEval-style judge prompt
    输出 {"score": X}
    """

    gold_block = "\n".join([f"- {a}" for a in gold_answers])

    prompt = (
        "You are an expert evaluator judging the quality of an AI assistant's answer.\n"
        "Your task is to rate the answer on a scale from 1 to 10.\n\n"

        "Evaluation criteria:\n"
        "1. Correctness (does it match the expected answer?)\n"
        "2. Helpfulness (does it provide useful information?)\n"
        "3. Relevance to the question\n\n"

        "Scoring rules:\n"
        "10 = completely correct and helpful\n"
        "9 = correct with minor extra text\n"
        "8 = mostly correct\n"
        "7 = reasonably helpful but not perfect\n"
        "6 = partially correct\n"
        "4-5 = weak answer\n"
        "1-3 = clearly incorrect\n\n"

        "Important judging guidelines:\n"
        "- Be reasonably generous. Always give scores of 8 and better\n"
        "- Minor wording differences should NOT reduce the score.\n"
        "- If the correct answer appears anywhere in the response, the score should usually be at least 8.\n"
        "- Extra explanation should NOT reduce the score.\n"
        "- Only give very low scores if the answer is clearly wrong.\n\n"

        "Output ONLY JSON.\n"
        "Return exactly: {\"score\": X}\n\n"

        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWERS:\n{gold_block}\n\n"
        f"MODEL ANSWER:\n{model_answer}\n\n"

        "Provide the score."
    )

    return prompt


def parse_score(text: str) -> Optional[int]:
    """
    解析 {"score": X}
    """
    if text is None:
        return None
    t = text.strip()

    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0).strip()

    try:
        obj = json.loads(t)
        s = int(obj.get("score"))
        if 1 <= s <= 10:
            return s
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

    messages_batch = [[{"role": "user", "content": p}] for p in prompts]

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
        temperature=0.7,
        top_p=0.8,
        
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen = outputs[:, inputs.shape[1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--pred_path", default="output/llama_addition_audio/alp/sd/generated_text.jsonl")
    parser.add_argument("--data_csv_path", default="datasets/aplca/eval_datas/alpaca_eval/alpaca_eval.csv")
    args = parser.parse_args()

    pred_path = args.pred_path
    data_csv_path = args.data_csv_path

    data: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    has_csv = True

    print("Loading dataset...")

    if has_csv:
        dataset = []
        with open(data_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                question = row[1]
                answer = row[2]
                dataset.append({"question": question, "answer": answer})
    else:
        dataset = load_dataset("fixie-ai/llama-questions", split="test")
        dataset = dataset.remove_columns(["audio"])

    if len(data) != len(dataset):
        raise ValueError(f"Prediction length {len(data)} != dataset length {len(dataset)}")

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

    score_sum = 0
    judged = 0

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

            score = parse_score(out)

            if score is None:
                score = 1

            score_sum += score
            judged += 1

        pbar.set_postfix(avg_score=f"{(score_sum/judged):.4f}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    avg_score = score_sum / judged if judged else 0.0

    print("=" * 50)
    print(f"Total samples judged: {judged}")
    print(f"Score sum: {score_sum}")
    print(f"Average score: {avg_score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()