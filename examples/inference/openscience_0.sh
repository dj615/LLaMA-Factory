export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12356

llamafactory-cli train examples/inference/llama3_predict_openscience_0.yaml
python examples/evaluate.py --task openscience --pred_file /root/workspace/data/predictions/openscience_predict_0/generated_predictions.jsonl
