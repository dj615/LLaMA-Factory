export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12355

llamafactory-cli train examples/train_full/llama3_full_sft_gsm8k_0.yaml

llamafactory-cli train examples/train_full/llama3_full_sft_gsm8k_1.yaml

llamafactory-cli train examples/train_full/llama3_full_sft_gsm8k_2.yaml