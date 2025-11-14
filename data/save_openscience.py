from datasets import load_dataset
import os
import json
import re

def extract_boxed(answer: str):
    """
    从字符串中提取 \boxed{...} 内容
    若不存在，返回 None
    """
    match = re.search(r"\\boxed\{(.+?)\}", answer)
    return match.group(1).strip() if match else None


# 1. 加载 HuggingFace 数据
dataset = load_dataset("nvidia/openscience", split="train")

# 2. 输出路径
train_path = "data/train/openscience.jsonl"
test_path = "data/test/openscience.jsonl"

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# 3. 写入 train（前5000条）
with open(train_path, "w", encoding="utf-8") as f_train:
    for item in dataset[:5000]:
        f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

# 4. 写入 test（5000~6000，增加 groundtruth）
with open(test_path, "w", encoding="utf-8") as f_test:
    for item in dataset[5000:6000]:
        new_item = {
            "input": item["input"],
            "output": item["output"],
            "groundtruth": extract_boxed(item["output"]),
        }
        f_test.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print("Done! Files saved:")
print(f" - {train_path}    (0~4999)")
print(f" - {test_path}     (5000~5999 with groundtruth)")
