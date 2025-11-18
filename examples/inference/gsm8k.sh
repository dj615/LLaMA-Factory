export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12355

llamafactory-cli train examples/inference/llama3_predict_gsm8k_0.yaml
python examples/evaluate.py --task gsm8k --pred_file /root/workspace/data/predictions/gsm8k_predict_0/generated_predictions.jsonl

llamafactory-cli train examples/inference/llama3_predict_gsm8k_1.yaml
python examples/evaluate.py --task gsm8k --pred_file /root/workspace/data/predictions/gsm8k_predict_1/generated_predictions.jsonl

llamafactory-cli train examples/inference/llama3_predict_gsm8k_2.yaml
python examples/evaluate.py --task gsm8k --pred_file /root/workspace/data/predictions/gsm8k_predict_2/generated_predictions.jsonl
