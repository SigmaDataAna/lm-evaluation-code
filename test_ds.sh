
export HF_ALLOW_CODE_EVAL=1

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=deepseek-ai/deepseek-coder-1.3b-instruct \
    --tasks humaneval \
    --batch_size 32 \
    --confirm_run_unsafe_code \
    --log_samples \
    --output_path ../outputs
