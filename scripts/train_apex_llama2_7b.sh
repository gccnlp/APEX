MODEL_PATH="${MODEL_PATH:-./models/Llama-2-7b-hf}"
DATA_PATH="${DATA_PATH:-./data/tulu_v2_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/apex_llama2_7b_k18.75_t2}"
PROJECT_NAME="${PROJECT_NAME:-apex}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

ZERO_STAGE=2
mkdir -p "$OUTPUT_DIR"

master_port=$((RANDOM % 5000 + 8001)) # add offload add master_port if socket error

deepspeed --include=localhost:0,1,2,3,4,5,6,7 \
    --master_port $master_port ./train/finetune.py \
    --model_name_or_path "$MODEL_PATH" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --max_seq_len 8192 \
    --learning_rate 2e-5 \
    --shuffle_dataloader \
    --shuffle_infer_data \
    --weight_decay 0. \
    --act_record_method "rank" \
    --mixer_lr_scale 1 \
    --lambda_reg 0 \
    --num_train_epochs 2 \
    --dtype bf16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --merge_to_base_model \
    --warmup_ratio 0.03 \
    --seed 42 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --gradient_checkpointing \
    --instruction_type single \
    --apex \
    --num_samples 12288 \
    --record_act_step 500 \
    --v_ratio 0.375 \
    --u_ratio 0.375 \
    --tuned_attn_ratio 0.5625 \
    --tuned_ffn_ratio 0.465 \
    --init_method "random" \
    --finetune_qk \
    --train_embedding_and_lm_head \
    --data_path "$DATA_PATH" \
    --project_name "$PROJECT_NAME" \
    --output_dir "$OUTPUT_DIR" 2> >(tee "$OUTPUT_DIR/err.log" >&2) | tee "$OUTPUT_DIR/training.log"
