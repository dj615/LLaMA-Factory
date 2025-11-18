from datasets import load_dataset
import os
import json

def extract_groundtruth(answer: str):
    """
    从 GSM8K 的 answer 字段中提取 #### 之后的最终答案。
    若不存在，则返回 None。
    """
    if "####" not in answer:
        return None
    return answer.split("####")[-1].strip()

train_set = load_dataset("gsm8k", "main", split="train")
test_set = load_dataset("gsm8k", "main", split="test")

train_dir = "/root/workspace/data/train"
train_file = os.path.join(train_dir, "gsm8k.jsonl")
os.makedirs(train_dir, exist_ok=True)

with open(train_file, "w", encoding="utf-8") as f:
    for example in train_set:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Saved HF gsm8k train to {train_file}")

test_dir = "/root/workspace/data/test"
test_file = os.path.join(test_dir, "gsm8k.jsonl")
os.makedirs(test_dir, exist_ok=True)

with open(test_file, "w", encoding="utf-8") as f:
    for example in test_set:
        new_item = {
            "question": example["question"],
            "answer": example["answer"],
            "groundtruth": extract_groundtruth(example["answer"]),
        }
        f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"Saved HF gsm8k test (with groundtruth) to {test_file}")
