
export HF_ALLOW_CODE_EVAL=1
export OUTPUT_DIR="/mnt/blob-shuailu1-tianyu-out/evaluation_results/hf_iter_220000"

lm_eval \
    --model hf \
    --model_args pretrained="/home/chentianyu/ckps/hf_iter_220000",dtype="bfloat16" \
    --task humaneval \
    --batch_size 256 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --log_samples \
    --output_path ${OUTPUT_DIR}

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval \
#     --model hf \
#     --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
#     --task HumanEval-MultiLineInfillingLight \
#     --batch_size 256 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code \
#     -o "${HUGGINGFACE_CKPT_PATH}/eval_mscode_multiline.json" \
#     2>&1 | tee "${HUGGINGFACE_CKPT_PATH}/eval_mscode_multiline.txt"
