
export HF_ALLOW_CODE_EVAL=1

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=ByteDance-Seed/Seed-Coder-8B-Instruct \
    --task livecodebench_instruct \
    --batch_size 64 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --log_samples \
    --output_path ../tianyu_outputs
