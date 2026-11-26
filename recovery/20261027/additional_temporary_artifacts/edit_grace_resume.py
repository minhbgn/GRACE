from pathlib import Path
p=Path('reproduction/GRACE/qwen-vl-finetune/scripts/finetune_qwen3vl_2b_grace_lora_ngpu.sh')
s=p.read_text().replace('SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}', 'SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}')
s=s.replace('DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-', 'TRAINING_BACKEND=${TRAINING_BACKEND:-zero2}\nDEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-',1)
s=s.replace('export HF_HOME=', '''BACKEND_ARGS=()
case "${TRAINING_BACKEND}" in
    ddp) ;;
    zero2) BACKEND_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}") ;;
    *) echo "ERROR: TRAINING_BACKEND must be ddp or zero2." >&2; exit 1 ;;
esac

export HF_HOME=''',1)
s=s.replace('    --deepspeed "${DEEPSPEED_CONFIG}" \\', '    "${BACKEND_ARGS[@]}" \\',1)
# Capture resolved settings before MODEL_NAME_OR_PATH is rewritten to checkpoint.
start=s.index('# Intermediate checkpoint directories')
s=s[:start]+'''# Store the resolved training recipe in every checkpoint. GPU count and
# accumulation may change together; model/data/optimizer/schedule must match.
RECIPE_KEYS=(TRAINING_BACKEND BASE_MODEL_DIR MODEL_NAME_OR_PATH TEACHER_MODEL_PATH
    DATASETS SHAREGPT4V_ROOT QAT_BITS QAT_GROUP_SIZE QAT_SCALE_LR LORA_R LORA_ALPHA
    LORA_DROPOUT LORA_TARGET_MODULES DKD_TEMPERATURE DKD_ALPHA DKD_BETA
    DKD_CHUNK_SIZE DKD_CHECKPOINT_CHUNKS RCKA_WEIGHT RCKA_LAYER IB_TAU IB_ETA
    IB_BETA_INIT IB_BETA_MIN IB_BETA_MAX DKD_WARMUP_STEPS RCKA_WARMUP_STEPS
    LEARNING_RATE MM_PROJECTOR_LR WEIGHT_DECAY WARMUP_RATIO MAX_GRAD_NORM
    NUM_TRAIN_EPOCHS MAX_STEPS MODEL_MAX_LENGTH MAX_PIXELS MIN_PIXELS PER_DEVICE_BATCH)
RECIPE_ARGS=()
for name in "${RECIPE_KEYS[@]}"; do RECIPE_ARGS+=("${name}=${!name}"); done
export GRACE_LAUNCH_RECIPE="${OUTPUT_DIR}/launch_recipe.json"
python "${SCRIPT_DIR}/grace_resume_recipe.py" "${OUTPUT_DIR}" "${NUM_GPUS}" \\
    "$((denominator * GRAD_ACCUM))" "${DEEPSPEED_CONFIG}" "${RECIPE_ARGS[@]}"

'''+s[start:]
s=s.replace('# so Trainer can resume optimizer/scheduler/model state exactly.', '# so Trainer can restore optimizer/scheduler/model state.')
s=s.replace('CONFIG_PATH="${OUTPUT_DIR}/config.json"', '''if [ -f "${OUTPUT_DIR}/PAUSED_AT_CHECKPOINT" ]; then
    echo "PAUSED: $(cat "${OUTPUT_DIR}/PAUSED_AT_CHECKPOINT")"
    echo "Restart this launcher with the same OUTPUT_DIR and TRAINING_BACKEND."
    exit 0
fi

CONFIG_PATH="${OUTPUT_DIR}/config.json"''')
s=s.replace('echo "  final_model=${OUTPUT_DIR}"', 'echo "  final_model=${OUTPUT_DIR}"\necho "  backend=${TRAINING_BACKEND}; to pause: touch ${OUTPUT_DIR}/STOP_AFTER_CHECKPOINT"')
p.write_text(s)
p=Path('reproduction/GRACE/qwen-vl-finetune/qwenvl/train/train_qwen_grace.py')
s=p.read_text().replace('from qwenvl.train.grace_profiler import (','from qwenvl.train.grace_resume import CheckpointPauseCallback\nfrom qwenvl.train.grace_profiler import (')
s=s.replace('    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):', '    pause_callback = CheckpointPauseCallback()\n    trainer.add_callback(pause_callback)\n\n    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):')
s=s.replace('    if grace_args.skip_final_export:', '    if pause_callback.paused:\n        rank0_print("[GRACE] Paused after checkpoint; skipping final merged export.")\n        _close_distributed_process_group()\n        return\n\n    if grace_args.skip_final_export:')
p.write_text(s)
