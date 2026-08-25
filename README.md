# APEX

Here is the implementation of our EMNLP 2026 Paper Advantageous Parameter Expansion Training Makes Better Large Language Models
<img width="816" height="325" alt="image" src="https://github.com/user-attachments/assets/e3616d84-ef1a-4c8d-9114-552585a5ed0a" />

## Contents

- `train/finetune.py`: stage-wise APEX instruction-tuning pipeline.
- `utils/apex_utils.py`: activation recording, ranking, conversion, and fusion.
- `apex_core/`: APEX linear layers and Monarch operators.
- `transformers/`: the project-specific Transformers 4.43.4 fork required for
  attention-head activation recording.
- `scripts/`: LLaMA-2 training configuration.

## Installation

```bash
pip install -r requirements.txt
pip install -e ./transformers
```

## Training

The script uses repository-relative defaults. Override `MODEL_PATH`,
`DATA_PATH`, `OUTPUT_DIR`, or `PROJECT_NAME` when needed, then run:

```bash
bash scripts/train_apex_llama2_7b.sh
```
