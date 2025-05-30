
export HF_ALLOW_CODE_EVAL=1

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=../hf_iter_10000/ \
    --tasks humaneval_instruct_sigma \
    --batch_size 32 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --log_samples \
    --output_path ../outputs
