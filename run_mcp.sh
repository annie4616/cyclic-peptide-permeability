#!/usr/bin/env bash
# Wrapper to launch a MultiCycPermea training run.
#
# Each invocation writes a private config directory so multiple runs can be
# launched in parallel without stepping on each other's YAML files.
#
# Env vars:
#   SPLIT        one of ID, OD, Cliff  (default ID)
#   TARGET       PAMPA | Permeability  (default PAMPA)
#   EPOCHS       override epoch count  (default 200)
#   BATCH_SIZE   override batch size   (default 16)
#   RUN_NAME     run name             (default derived)
#   WANDB_PROJECT default cyclic-peptide-permeability
#   USE_AUTHOR_SPLIT  1 -> use author's data/remove_strange_values split
#   CUDA_VISIBLE_DEVICES  default "0,1"

set -euo pipefail

ROOT=/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability
MCP="$ROOT/MultiCycPermea"
VENV="$ROOT/mcp_env"
DL="$MCP/DL"

SPLIT=${SPLIT:-ID}
TARGET=${TARGET:-PAMPA}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-16}
RUN_NAME=${RUN_NAME:-"${SPLIT}_${TARGET}_$(date +%Y%m%d_%H%M%S)"}
WANDB_PROJECT=${WANDB_PROJECT:-cyclic-peptide-permeability}
USE_AUTHOR_SPLIT=${USE_AUTHOR_SPLIT:-0}

# --- Pick split CSVs --------------------------------------------------------
if [[ "$USE_AUTHOR_SPLIT" == "1" ]]; then
    TRAIN_PATH="data/remove_strange_values/train.csv"
    VAL_PATH="data/remove_strange_values/val.csv"
    TEST_PATH="data/remove_strange_values/test.csv"
else
    TRAIN_PATH="data/ours/${SPLIT}_train.csv"
    VAL_PATH="data/ours/${SPLIT}_val.csv"
    TEST_PATH="data/ours/${SPLIT}_test.csv"
fi

# --- Per-run config dir (so parallel runs don't clobber each other) ---------
RUN_CFG="$DL/config_runs/$RUN_NAME"
mkdir -p "$RUN_CFG"

cat > "$RUN_CFG/model.yaml" <<YAML
data_yaml:
  text_data_yaml: "${RUN_CFG}/smi_dataset.yaml"
  image_data_yaml: "${RUN_CFG}/img_dataset.yaml"
model_yaml:
  text_model_yaml: "${RUN_CFG}/Transformer.yaml"
  image_model_yaml: "${RUN_CFG}/TIMM.yaml"
  use_text_info: True
  use_image_info: True
  use_fingerprint_info: False
  feature_cmb_type: concate
YAML

cat > "$RUN_CFG/smi_dataset.yaml" <<YAML
train_data_path: '${TRAIN_PATH}'
test_data_path: '${TEST_PATH}'
val_data_path: '${VAL_PATH}'
data_type: "SMILES"
target: "${TARGET}"
max_len: 250
augmentation: False
augmentation_ratio: 0.5
load_vocab: True
vocab_path: '../data/vocab.pkl'
use_HELM: False
YAML

cat > "$RUN_CFG/img_dataset.yaml" <<YAML
image_folder: "data/cycle_peptide/cycle_peptide_image_png"
augmentation: True
YAML

cat > "$RUN_CFG/Transformer.yaml" <<YAML
model_type: "Transformer"
epochs: ${EPOCHS}
lr : 0.0001
batch_size: ${BATCH_SIZE}
emb_dim: 256
hidden_dim: 256
output_dim: 1
num_head: 8
num_layers: 8
drop_out: 0.1
YAML

cat > "$RUN_CFG/TIMM.yaml" <<YAML
model_type: "swin_molscribe"
image_size: 384
pretrained: True
lr : 0.0001
YAML

# --- CPU / thread caps ------------------------------------------------------
# Caller may pass CPU_AFFINITY="0-63", "64-127", etc. to partition the
# 128-core budget (50% of this 256-core server) across parallel runs.
CPU_AFFINITY=${CPU_AFFINITY:-0-127}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}

# --- GPU / wandb ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export RUN_NAME
export WANDB_PROJECT
if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
    echo "[run_mcp] No WANDB_API_KEY and no ~/.netrc; running WANDB_MODE=offline"
    export WANDB_MODE=offline
fi

echo "============================================================"
echo " MultiCycPermea run: $RUN_NAME"
echo "   SPLIT=$SPLIT   TARGET=$TARGET   EPOCHS=$EPOCHS   BATCH=$BATCH_SIZE"
echo "   USE_AUTHOR_SPLIT=$USE_AUTHOR_SPLIT"
echo "   GPUs=$CUDA_VISIBLE_DEVICES   CPU affinity=$CPU_AFFINITY   BLAS threads=$OMP_NUM_THREADS"
echo "   WANDB_PROJECT=$WANDB_PROJECT   WANDB_MODE=${WANDB_MODE:-online}"
echo "   CFG=$RUN_CFG"
echo "============================================================"

cd "$DL"
exec taskset -c "$CPU_AFFINITY" "$VENV/bin/python" main.py --all_config "$RUN_CFG/model.yaml"
