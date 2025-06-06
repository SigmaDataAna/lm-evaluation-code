
export HF_ALLOW_CODE_EVAL=1
mkdir -p ${SAVE_PATH}

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Base,dtype="bfloat16",parallelize=True \
    --task mbpp \
    --batch_size 16 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    --log_samples \
    --output_path "${HUGGINGFACE_CKPT_PATH_BASE}/eval/coderv2lite_eval_mbpp"


# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Base,dtype="bfloat16",parallelize=True \
#     --task HumanEval-MultiLineInfillingLight-DS \
#     --batch_size 256 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code \
#     -o "${HUGGINGFACE_CKPT_PATH_BASE}/eval/coderv2lite_eval_mscode_multiline" \
#     2>&1 | tee "${SAVE_PATH}/coderv2lite_eval_mscode_multiline.txt"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Base,dtype="bfloat16",parallelize=True \
#     --task HumanEval-RandomSpanInfillingLight-DS \
#     --batch_size 256 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code \
#     -o "${HUGGINGFACE_CKPT_PATH_BASE}/eval/coderv2lite_eval_mscode_randomspan" \
#     2>&1 | tee "${SAVE_PATH}/coderv2lite_eval_mscode_randomspan.txt"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Base,dtype="bfloat16",parallelize=True \
#     --task HumanEval-SingleLineInfilling-DS \
#     --batch_size 256 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code \
#     -o "${HUGGINGFACE_CKPT_PATH_BASE}/eval/coderv2lite_eval_mscode_singleline" \
#     2>&1 | tee "${SAVE_PATH}/coderv2lite_eval_mscode_singleline.txt"
