# export CUDA_VISIBLE_DEVICES=0

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/opt/conda/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path (Data preprocessing comming soon!)
TRAIN_SET_PATH="/mnt/csi-data-aly/shared/public/ruiwan/diffusion_planner_data"
TRAIN_SET_LIST_PATH="./diffusion_planner_training.json"
###################################

# $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone train_predictor.py \
# --train_set  $TRAIN_SET_PATH \
# --train_set_list  $TRAIN_SET_LIST_PATH \


if [ $CUDA_VISIBLE_DEVICES ]; then
    IFS=','
    read -a GPUS <<<"$CUDA_VISIBLE_DEVICES"
    echo 'use gpus: '${!GPUS[@]}
    GPUS=${#GPUS[@]}
else
    GPUS=$(nvidia-smi -L | wc -l)
    echo 'use all '$GPUS' gpus in this machine'
fi

if [ -n "$MLP_WORKER_NUM" ]; then
    echo "Running on HUOSHAN YUN ..."
    torchrun --nproc_per_node=$MLP_WORKER_GPU \
            --nnodes=$MLP_WORKER_NUM \
            --node_rank=$MLP_ROLE_INDEX \
            --master_addr=$MLP_WORKER_0_HOST \
            --master_port=$MLP_WORKER_0_PORT \
            train_predictor.py --train_set  $TRAIN_SET_PATH \
            --train_set_list  $TRAIN_SET_LIST_PATH 
elif [ -z $PAI_TASK_ROLE_TASK_COUNT_taskrole ]; then
    if [ -z ${WORLD_SIZE} ]; then
        torchrun --nproc_per_node=${GPUS} train_predictor.py --train_set  $TRAIN_SET_PATH \
            --train_set_list  $TRAIN_SET_LIST_PATH \
            "$@"
    else
        if [ $SG_PAI ]; then
            echo "Running on SG-PAI ..."
            RANK=$(echo $(hostname) | sed 's/worker-\([0-9]*\)/\1/g')
            echo ' Current Node Rank: ' ${RANK}
        else
            echo "Running on PAI ..."
        fi
        torchrun --nproc_per_node=${GPUS} \
            --nnodes=${WORLD_SIZE} \
            --node_rank=${RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            train_predictor.py --train_set  $TRAIN_SET_PATH \
            --train_set_list  $TRAIN_SET_LIST_PATH \
            "$@"

    fi
else
    torchrun --nproc_per_node=${GPUS} \
        --nnodes=$PAI_TASK_ROLE_TASK_COUNT_taskrole \
        --node_rank=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX \
        --master_addr=$PAI_HOST_IP_taskrole_0 \
        --master_port=$PAI_taskrole_0_SynPort_PORT \
        train_predictor.py --train_set  $TRAIN_SET_PATH \
            --train_set_list  $TRAIN_SET_LIST_PATH \
            "$@"

fi

