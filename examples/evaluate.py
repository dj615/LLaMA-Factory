import json
import os
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "openscience"])
    parser.add_argument("--pred_file", type=str, required=True,
                        help="模型预测结果文件（json 或 jsonl）")
    return parser.parse_args()


def extract_after_hashes(text):
    """用于 GSM8K：提取 #### 后内容"""
    if not isinstance(text, str):
        return None
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


def extract_boxed(text):
    """用于 OpenScience：提取 \\boxed{...}"""
    if not isinstance(text, str):
        return None
    m = re.search(r"\\boxed\{(.+?)\}", text)
    return m.group(1).strip() if m else None


def load_jsonl(path):
    """读取 jsonl 文件为列表"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    args = parse_args()
    task = args.task

    # 自动决定 groundtruth 文件
    gt_file = f"data/test/{task}.jsonl"
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"groundtruth 文件不存在: {gt_file}")

    print(f"Using groundtruth file: {gt_file}")

    # 1. 使用传入的 pred_file，不再搜索
    pred_path = args.pred_file
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    print(f"Using prediction file: {pred_path}")

    # 2. 读取 prediction
    if pred_path.endswith(".jsonl"):
        preds = load_jsonl(pred_path)
    else:
        preds = json.load(open(pred_path, "r", encoding="utf-8"))
        if isinstance(preds, dict) and "predictions" in preds:
            preds = preds["predictions"]

    # 3. 提取预测答案
    model_answers = []
    for p in preds:
        if isinstance(p, dict):
            text = p.get("prediction") or p.get("text") or p.get("output") or str(p)
        else:
            text = str(p)

        if task == "gsm8k":
            ans = extract_after_hashes(text)
        else:  # openscience
            ans = extract_boxed(text)

        model_answers.append(ans)

    print(f"Loaded {len(model_answers)} predicted answers.")

    # 4. 加载 groundtruth
    gt_data = load_jsonl(gt_file)
    groundtruths = [item["groundtruth"] for item in gt_data]
    print(f"Loaded {len(groundtruths)} groundtruth answers.")

    # 5. 对齐
    n = min(len(model_answers), len(groundtruths))
    print(f"Evaluating first {n} samples...")

    # 6. 计算 accuracy
    correct = 0
    for i in range(n):
        pred = model_answers[i]
        gt = str(groundtruths[i]).strip()

        if pred is None:
            continue

        if pred.strip() == gt:
            correct += 1

    acc = correct / n if n > 0 else 0
    print(f"\nAccuracy: {acc:.4f}  ({correct}/{n})")


if __name__ == "__main__":
    main()
