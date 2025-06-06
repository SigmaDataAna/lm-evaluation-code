
export HF_ALLOW_CODE_EVAL=1
mkdir -p "${HUGGINGFACE_CKPT_PATH_BASE}/eval"

ckps=()
current=220000
incr_first=150
incr_next=350
count=5

for ((i=0; i<count; i++))
do
    ckps+=("hf_iter_${current}")
    if [ $i -eq 0 ]; then
        current=$((current + incr_first))
    else
        current=$((current + incr_next))
    fi
done

for ckp in "${ckps[@]}"
do
    HUGGINGFACE_CKPT_PATH="${HUGGINGFACE_CKPT_PATH_BASE}/${ckp}"
    echo "Evaluating checkpoint: ${HUGGINGFACE_CKPT_PATH}"
    cp test_scripts/tokenization_sigma.py "${HUGGINGFACE_CKPT_PATH}/tokenization_sigma.py"

    accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
        --task humaneval \
        --batch_size 64 \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        -o "${HUGGINGFACE_CKPT_PATH_BASE}/eval/eval_humaneval_${ckp}.json" \
        2>&1 | tee "${HUGGINGFACE_CKPT_PATH_BASE}/eval/eval_humaneval_${ckp}.txt"

    accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
        --task mbpp \
        --batch_size 64 \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        -o "${HUGGINGFACE_CKPT_PATH_BASE}/eval/eval_mbpp_${ckp}.json" \
        2>&1 | tee "${HUGGINGFACE_CKPT_PATH_BASE}/eval/eval_mbpp_${ckp}.txt"

    accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained="${HUGGINGFACE_CKPT_PATH}",dtype="bfloat16",parallelize=True \
        --task HumanEval-MultiLineInfillingLight \
        --batch_size 64 \
        --confirm_run_unsafe_code \
        --trust_remote_code \
        -o "${HUGGINGFACE_CKPT_PATH_BASE}/eval/eval_msinternal_multiline_light_${ckp}.json" \
        2>&1 | tee "${HUGGINGFACE_CKPT_PATH_BASE}/eval/eval_msinternal_multiline_light_${ckp}.txt"
done