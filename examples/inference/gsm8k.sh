export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=12355

llamafactory-cli train examples/inference/llama3_predict_gsm8k_0.yaml
llamafactory-cli train examples/inference/llama3_predict_gsm8k_1.yaml
llamafactory-cli train examples/inference/llama3_predict_gsm8k_2.yaml