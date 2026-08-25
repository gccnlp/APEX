# APEX

This repository provides the instruction-tuning implementation of APEX,
including activation-based advantage assessment, stage-wise expansion
training, and operator fusion.

## Contents

- `train/finetune.py`: stage-wise APEX instruction-tuning pipeline.
- `utils/apex_utils.py`: activation recording, ranking, conversion, and fusion.
- `apex_core/`: APEX linear layers and Monarch operators.
- `transformers/`: the project-specific Transformers 4.43.4 fork required for
  attention-head activation recording.
- `scripts/`: LLaMA-2 training configuration.

Some internal parameters retain the name `s2`; these implement the direct
parameter updates used by APEX.

## Installation

```bash
pip install -r requirements.txt
pip install -e ./transformers
```

Do not replace the included Transformers fork with the stock 4.43.4 package.
The APEX attention statistics rely on modifications in the included model
implementations.

## Training

The script uses repository-relative defaults. Override `MODEL_PATH`,
`DATA_PATH`, `OUTPUT_DIR`, or `PROJECT_NAME` when needed, then run:

```bash
bash scripts/train_apex_llama2_7b.sh
```

The LLaMA-2 configuration uses `T=2` and an initial threshold of `K=18.75%`.
