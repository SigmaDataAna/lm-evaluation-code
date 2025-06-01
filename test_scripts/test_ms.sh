
export HF_ALLOW_CODE_EVAL=1

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --task HumanEval-MultiLineInfillingLight \
    --batch_size 64 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --log_samples \
    --output_path ../tianyu_outputs
