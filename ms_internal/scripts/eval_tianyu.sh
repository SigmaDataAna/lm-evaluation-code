#!/bin/bash
model_url="http://localhost:8000/v1" # model_url
api_key="EMPTY" # api_key
api_type="cont" # chat or fim or cont
benchmark="multi-line-light" # "test", "single-line", "multi-line", "random-span", "random-span-light"
model_output_file="../../tianyu_outputs/model_output.json" # model output
eval_output_file="../../tianyu_outputs/eval_output.json" # console output

start_time=$(date +%s)
python scripts/data_gen.py --model_url=$model_url --api_key=$api_key --api_type=$api_type --benchmark=$benchmark --output_file=$model_output_file
# evaluate_infilling_functional_correctness $model_output_file --benchmark_name=$benchmark > $eval_output_file
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: ${elapsed_time} seconds"


# curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
#         "messages": [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "Who won the world series in 2020?"}
#         ]
#     }'
