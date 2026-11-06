import argparse
import json
import re
from datasets import load_dataset
from tqdm import tqdm


def normalize(text: str) -> str:
    """
    标准化文本：
    - 转小写
    - 去标点
    - 去多余空格
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_answer_loose(pred_text: str, answer: str) -> bool:
    """
    宽松匹配：答案字符串包含于预测文本
    """
    return answer in pred_text


def contains_answer_strict(pred_text: str, answer: str) -> bool:
    """
    严格匹配：单词边界匹配
    避免 May 匹配 maybe
    """
    pattern = r"\b" + re.escape(answer) + r"\b"
    return re.search(pattern, pred_text) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default="output/ablation/no_target_near/generated_text.jsonl")
    args = parser.parse_args()
    data = []
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            #breakpoint()
            #print(line)
            data.append(json.loads(line))

    print("Loading dataset...")
    dataset = load_dataset("fixie-ai/trivia_qa-audio", split='validation')
    dataset = dataset.remove_columns(["question_audio"])
    

    if len(data) != len(dataset):
        raise ValueError(
            f"Prediction length {len(data)} "
            f"!= dataset length {len(dataset)}"
        )

    correct = 0
    for i in range(len(data)):
        if data[i]["text"] == "":
            correct+=1
            continue
        pred_text = normalize(data[i]['text'])
        gt_answers = dataset[data[i]['index']]["answer"]["aliases"]

        hit = False
        for ans in gt_answers:
            ans_norm = normalize(ans)

            
            if contains_answer_loose(pred_text, ans_norm):
                hit = True
                break

        if hit:
            correct += 1

    accuracy = correct / len(dataset)

    print("=" * 50)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()