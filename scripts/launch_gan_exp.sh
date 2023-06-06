#!/bin/bash
function output {
    eval ${cmd}
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo -e "\e[1;32m ${cmd} [Success] \e[0m"
    else
        echo -e "\e[1;31m ${cmd} [Failure] \e[0m"
        exit 1
    fi
}

LOG_DIR="/media/fangda/FDL2/gsan_exp/"
CFG_DIR="configs/projects/gsan"
#BASE_CMD="python train.py --single_gpu "
TRAIN_CMD="python -m torch.distributed.launch --nproc_per_node=1 train.py"
TEST_CMD="python -m torch.distributed.launch --nproc_per_node=1 inference.py"

## declare an array variable
declare -a experiments=(
    "gsan"
)

postfix=""

## now loop through the above array
for exp in "${experiments[@]}"; do
    cmd="${TRAIN_CMD} --config ${CFG_DIR}/${exp}.yaml --logdir ${LOG_DIR}/${exp}${postfix} --single_gpu >> ${LOG_DIR}/${exp}${postfix}_train.log "
    output
    sleep 5
done
