#!/bin/bash
# ./run.sh <script> <config.yaml> [--key.subkey=value ...]
#   *.sbatch.sh -> sbatch (logs/<script>.<config>_<jobid>.out/.err);  *.py -> python on this node.
# For slurm/sft.sbatch.sh the config's `run:` section (nodes, train_nodes, qos, time, exclude) sets
# the sbatch flags; --run.* overrides on the command line win, the rest go to the recipe.
ROOT_PATH=$(dirname "$(realpath "$0")")
cd "$ROOT_PATH"
source .env

RUN_FILE=$1
CONFIG=$2
LOG_FILE=$(basename "$RUN_FILE" .sh)
LOG_FILE=${LOG_FILE%.sbatch}.$(basename "$CONFIG" .yaml)
mkdir -p logs

if [[ "$RUN_FILE" == *.py ]]; then
    exec python "$RUN_FILE" "${@:2}"
fi

SBATCH_ARGS=()
if [[ "$RUN_FILE" == */sft.sbatch.sh ]]; then
    read -r NODES TRAIN_NODES QOS TIME EXCLUDE <<< "$(python - "${@:2}" <<'PY'
import sys, yaml
run = yaml.safe_load(open(sys.argv[1])).get("run") or {}
for arg in sys.argv[2:]:
    if arg.startswith("--run."):
        key, value = arg[len("--run."):].split("=", 1)
        run[key] = value
print(*(run.get(k) or "-" for k in ("nodes", "train_nodes", "qos", "time", "exclude")))
PY
)"
    [ "$NODES" != "-" ] && SBATCH_ARGS+=(--nodes="$NODES")
    [ "$QOS" != "-" ] && SBATCH_ARGS+=(--qos="$QOS")
    [ "$TIME" != "-" ] && SBATCH_ARGS+=(--time="$TIME")
    [ "$EXCLUDE" != "-" ] && SBATCH_ARGS+=(--exclude="$EXCLUDE")
    # train_nodes < nodes: allocate `nodes` (bprod minimum), train on the first `train_nodes`.
    [ "$TRAIN_NODES" != "-" ] && export TRAIN_NODES
fi

sbatch "${SBATCH_ARGS[@]}" --job-name="$LOG_FILE" \
    --output="logs/${LOG_FILE}_%j.out" --error="logs/${LOG_FILE}_%j.err" "$RUN_FILE" "${@:2}"
