
export HF_ALLOW_CODE_EVAL=1

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Base,dtype="bfloat16",parallelize=True \
    --task humaneval \
    --batch_size 256 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    2>&1 | tee "${SAVE_PATH}/coderv2lite_eval_humaneval.txt"


accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Base,dtype="bfloat16",parallelize=True \
    --task HumanEval-MultiLineInfillingLight-DS \
    --batch_size 256 \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    2>&1 | tee "${SAVE_PATH}/coderv2lite_eval_mscode_multiline.txt"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite,dtype="bfloat16",parallelize=True \
#     --task HumanEval-SingleLineInfilling \
#     --batch_size 256 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code \
#     2>&1 | tee "${SAVE_PATH}/v2lite_eval_mscode_singleline.txt"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite,dtype="bfloat16",parallelize=True \
#     --task HumanEval-MultiLineInfillingLight \
#     --batch_size 256 \
#     --confirm_run_unsafe_code \
#     --trust_remote_code \
#     2>&1 | tee "${SAVE_PATH}/v2lite_eval_mscode_multiline.txt"
