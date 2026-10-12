#!/bin/bash
# Several runs of one config side by side in a single allocation (e.g. bprod: >= 65 nodes).
# The config's run.sweep maps a run name to its recipe overrides; each run gets run.sweep_nodes
# nodes, its own rendezvous, checkpoint dir (<checkpoint_dir>/<name>) and wandb name
# (<wandb.name>_<name>). Nodes no run uses, and those of a run that has finished, are released;
# the first node (this script) stays, so allocate 1 + runs x sweep_nodes.
#   ./run.sh slurm/sweep.sbatch.sh configs/sft/minervav2_20b_64k.yaml
#SBATCH -A FAIR_NLP
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=494000M

CONFIG=$1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_ALLOC_CONF=expandable_segments:True
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / GPUS_PER_NODE ))
RDZV_TIMEOUT=${RDZV_TIMEOUT:-1200}
echo "Repo: $(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)  CONFIG=$CONFIG"

# name <TAB> overrides, one line per run, plus the per-run node count and the shared output names.
read -r NODES_PER_RUN CKPT_DIR WANDB_NAME <<< "$(python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["run"]["sweep_nodes"], c["checkpoint"]["checkpoint_dir"], c["wandb"]["name"])
PY
)"
mapfile -t RUNS < <(python - "$CONFIG" <<'PY'
import sys, yaml
for name, overrides in yaml.safe_load(open(sys.argv[1]))["run"]["sweep"].items():
    print(f"{name}\t{overrides}")
PY
)
# The first node runs this script and is never released; the runs use the nodes after it.
mapfile -t ALL < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
BATCH_HOST=${ALL[0]}
HOSTS=("${ALL[@]:1}")
(( ${#RUNS[@]} * NODES_PER_RUN <= ${#HOSTS[@]} )) || { echo "need 1 + ${#RUNS[@]} x $NODES_PER_RUN nodes"; exit 1; }

# Shrink the allocation to the nodes of the runs still going (Slurm resizes a running job in place).
release_all_but() {
    local keep
    keep=$(printf '%s\n' "$BATCH_HOST" "$@" | paste -sd,)
    scontrol update JobId="$SLURM_JOB_ID" NodeList="$keep" >/dev/null && echo "[$(date)] allocation now: $keep"
    rm -f "slurm_job_${SLURM_JOB_ID}_resize."{sh,csh}  # written by every resize; this script needs no env reset
}

declare -A PID_NAME PID_NODES
for i in "${!RUNS[@]}"; do
    IFS=$'\t' read -r NAME OVERRIDES <<< "${RUNS[$i]}"
    NODES=("${HOSTS[@]:$((i * NODES_PER_RUN)):$NODES_PER_RUN}")
    NODELIST=$(printf '%s\n' "${NODES[@]}" | paste -sd,)
    # IPv4 of the run's first node (a node with broken DNS once hung a 64-node rendezvous).
    ADDR=$(getent ahostsv4 "${NODES[0]}" | awk 'NR==1 {print $1}')
    PORT=$((29500 + i))
    echo "[$(date)] $NAME on $NODELIST (rdzv $ADDR:$PORT): $OVERRIDES"
    srun --export=ALL --nodes="$NODES_PER_RUN" --ntasks="$NODES_PER_RUN" --ntasks-per-node=1 --nodelist="$NODELIST" \
        --output="logs/${SLURM_JOB_NAME}.${NAME}_${SLURM_JOB_ID}.out" --error="logs/${SLURM_JOB_NAME}.${NAME}_${SLURM_JOB_ID}.err" \
        bash -c "torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NODES_PER_RUN --rdzv_backend=c10d \
            --rdzv_endpoint=$ADDR:$PORT --rdzv_conf=read_timeout=$RDZV_TIMEOUT,join_timeout=$RDZV_TIMEOUT \
            -m nemo_automodel.cli.app $CONFIG $OVERRIDES \
            --checkpoint.checkpoint_dir=$CKPT_DIR/$NAME --wandb.name=${WANDB_NAME}_$NAME" &
    PID_NAME[$!]=$NAME
    PID_NODES[$!]="${NODES[*]}"
done
sleep 60  # let every step start before giving nodes back
running_nodes() { for p in "${!PID_NAME[@]}"; do echo ${PID_NODES[$p]}; done | tr ' ' '\n'; }
mapfile -t KEEP < <(running_nodes)
release_all_but "${KEEP[@]}"

EXIT=0
while (( ${#PID_NAME[@]} )); do
    sleep 60  # bash 4.4 on the nodes: no `wait -n -p`, so poll the runs
    for DONE in "${!PID_NAME[@]}"; do
        kill -0 "$DONE" 2>/dev/null && continue
        wait "$DONE"
        STATUS=$?
        echo "[$(date)] ${PID_NAME[$DONE]} exited with status $STATUS"
        (( STATUS != 0 )) && EXIT=$STATUS
        unset "PID_NAME[$DONE]" "PID_NODES[$DONE]"
        mapfile -t KEEP < <(running_nodes)
        (( ${#PID_NAME[@]} )) && release_all_but "${KEEP[@]}"
    done
done
exit $EXIT
