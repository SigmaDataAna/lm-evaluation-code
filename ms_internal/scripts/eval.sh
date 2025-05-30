#!/bin/sh
model_url=$1 # model_url
api_key=$2 # api_key
api_type=$3 # chat or fim
benchmark=$4 # "test", "single-line", "multi-line", "random-span", "random-span-light"
model_output_file=$5 # model output
eval_output_file=$6 # console output

start_time=$(date +%s)
python3 data_gen.py --model_url=$model_url --api_key=$api_key --api_type=$api_type --benchmark=$benchmark --output_file=$model_output_file
evaluate_infilling_functional_correctness $model_output_file --benchmark_name=$benchmark > $eval_output_file
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: ${elapsed_time} seconds"