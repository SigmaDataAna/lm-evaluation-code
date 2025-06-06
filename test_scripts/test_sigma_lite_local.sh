
export HF_ALLOW_CODE_EVAL=1
mkdir -p "${HUGGINGFACE_CKPT_PATH}/eval"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
    --task humaneval \
    --batch_size 64 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    -o "${HUGGINGFACE_CKPT_PATH}/eval/eval_mscode_singleline.json" \
    2>&1 | tee "${HUGGINGFACE_CKPT_PATH}/eval/eval_mscode_singleline.txt"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
    --task mbpp \
    --batch_size 64 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    -o "${HUGGINGFACE_CKPT_PATH}/eval/eval_mscode_multiline.json" \
    2>&1 | tee "${HUGGINGFACE_CKPT_PATH}/eval/eval_mscode_multiline.txt"
