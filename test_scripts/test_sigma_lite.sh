
export HF_ALLOW_CODE_EVAL=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval \
    --model hf \
    --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
    --task HumanEval-SingleLineInfilling \
    --batch_size 256 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    -o "${HUGGINGFACE_CKPT_PATH}/eval_mscode_singleline.json" \
    2>&1 | tee "${HUGGINGFACE_CKPT_PATH}/eval_mscode_singleline.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval \
    --model hf \
    --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
    --task HumanEval-MultiLineInfillingLight \
    --batch_size 256 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    -o "${HUGGINGFACE_CKPT_PATH}/eval_mscode_multiline.json" \
    2>&1 | tee "${HUGGINGFACE_CKPT_PATH}/eval_mscode_multiline.txt"
