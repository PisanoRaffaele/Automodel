#!/bin/bash
# NeMo AutoModel SFT on Leonardo Booster (4x A100-64GB per node, no internet on compute nodes).
# Launch through run.sh (sources .env, sbatch flags from the config's `run:`):
#   ./run.sh slurm/sft.sbatch.sh configs/sft/minervav2_20b_4k_parity.yaml [--key.subkey=value ...]
#SBATCH -A FAIR_NLP
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000M               # all allocatable: 514000M - MemSpecLimit 20000M; bprod needs >= 31280G over 65 nodes
#SBATCH --signal=B:USR1@300

CONFIG=$1
OVERRIDES="${*:2}"  # recipe overrides (run.* ones are ignored by the recipe)
if [ -z "$CONFIG" ]; then
    echo "usage: ./run.sh slurm/sft.sbatch.sh <config.yaml> [--key.subkey=value ...]"
    exit 1
fi

# Offline HF: model/datasets must be pre-downloaded on a login node into $HF_HOME.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Launch geometry -- every knob overridable from the environment (run.sh exports TRAIN_NODES
# from the config's run.train_nodes). TRAIN_NODES < allocated nodes leaves the rest idle,
# e.g. a QOS that requires a minimum node count.
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
TRAIN_NODES=${TRAIN_NODES:-$SLURM_NNODES}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( ${SLURM_CPUS_PER_TASK:-32} / GPUS_PER_NODE ))}
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# IPv4 instead of the hostname: a node with broken name resolution (lrdn1209, 2026-10-09) could not
# resolve the master and hung the whole 64-node rendezvous.
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
MASTER_ADDR=${MASTER_ADDR:-$MASTER_HOST}
export MASTER_PORT=${MASTER_PORT:-29500}
# At 64 nodes the master's torchrun can come up minutes after the others (torch imports from
# Lustre), and the 60 s default made late connectors give up. Seconds, for connect and join.
RDZV_TIMEOUT=${RDZV_TIMEOUT:-1200}
echo "Hosts: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')  CONFIG=$CONFIG $OVERRIDES"
echo "Launch: ${TRAIN_NODES}/${SLURM_NNODES} nodes x ${GPUS_PER_NODE} GPUs, rdzv ${MASTER_ADDR}:${MASTER_PORT}"
echo "Repo: $(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)  venv=$UV_PROJECT_ENVIRONMENT"

# Forward SIGUSR1 to the training workers only (not srun/torchrun): every rank checkpoints at the
# next step boundary once any rank sees it (see the repo's slurm.sub for the full rationale).
forward_preemption_signal() {
    echo "[$(date)] SIGUSR1 from SLURM; forwarding to local training workers"
    for pid in $(pgrep -u "$(id -un)" -f "nemo_automodel.cli.app"); do
        if ! tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q torchrun; then
            kill -USR1 "${pid}" 2>/dev/null || true
        fi
    done
}
trap forward_preemption_signal USR1

# One torchrun per node (same launch as the repo's slurm.sub); each spawns GPUS_PER_NODE workers.
# srun takes the first TRAIN_NODES nodes of the allocation, which includes MASTER_ADDR.
srun --export=ALL --nodes="${TRAIN_NODES}" --ntasks="${TRAIN_NODES}" --ntasks-per-node=1 bash -c "\
    torchrun \
        --nproc_per_node=${GPUS_PER_NODE} \
        --nnodes=${TRAIN_NODES} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv_conf=read_timeout=${RDZV_TIMEOUT},join_timeout=${RDZV_TIMEOUT} \
        -m nemo_automodel.cli.app ${CONFIG} ${OVERRIDES}" &
SRUN_PID=$!
wait ${SRUN_PID}
trap '' USR1
wait ${SRUN_PID}
EXIT_CODE=$?
echo "[$(date)] srun exited with status ${EXIT_CODE}"
exit ${EXIT_CODE}
