
export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES="2,3"

lm_eval --model hf \
    --model_args pretrained=/mnt/hdd1/huggingfacecache/hf_cache_lizongyang/models--Qwen--Qwen3-0.6B/snapshots/6130ef31402718485ca4d80a6234f70d9a4cf362 \
    --tasks HumanEval-MultiLineInfillingLight \
    --batch_size 8 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --log_samples \
    --output_path ../outputs
