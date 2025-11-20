import json
import os
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "openscience"])
    parser.add_argument("--pred_file", type=str, required=True, help="模型预测结果文件（json 或 jsonl），包含 prompt/predict/label")
    return parser.parse_args()


def norm(x: str):
    """轻微 normalization"""
    if x is None:
        return None
    return x.strip().lower()


def extract_after_hashes(text):
    """用于 GSM8K：提取 #### 后内容"""
    if not isinstance(text, str):
        return None
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def extract_boxed(text):
    """用于 OpenScience：提取 \boxed{...}（注意不是 \\boxed）"""
    if not isinstance(text, str):
        return None
    m = re.search(r"\\boxed\{(.+?)\}", text)
    return m.group(1).strip() if m else None


def extract_answer(text, task):
    """自动按任务提取答案"""
    if task == "gsm8k":
        return extract_after_hashes(text)
    else:
        return extract_boxed(text)


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    args = parse_args()
    task = args.task
    pred_path = args.pred_file

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    print(f"Using prediction file: {pred_path}")

    # 读取 prediction 文件
    if pred_path.endswith(".jsonl"):
        data = load_jsonl(pred_path)
    else:
        data = json.load(open(pred_path, "r", encoding="utf-8"))

    print(f"Loaded {len(data)} records\n")

    correct = 0
    total = 0

    for item in data:
        predict_raw = item.get("predict", "")
        label_raw = item.get("label", "")

        # 提取答案
        pred_ans = extract_answer(predict_raw, task)
        gt_ans = extract_answer(label_raw, task)

        # 如果 gt 没提取到，直接用 label 原文
        if gt_ans is None:
            gt_ans = str(label_raw).strip()

        pred_ans = norm(pred_ans)
        gt_ans = norm(gt_ans)

        total += 1

        if pred_ans is None:  # predict 无法提取 → 判错
            continue

        if pred_ans == gt_ans:
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    main()
