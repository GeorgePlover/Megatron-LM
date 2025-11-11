#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/home/nsh/nas/projects/Megatron-LM/examples/gpt3/checkpoint #<Specify path>
TENSORBOARD_LOGS_PATH=/home/nsh/nas/projects/Megatron-LM/examples/gpt3/tensorboard_logs #<Specify path>
VOCAB_FILE=/home/nsh/nas/projects/Megatron-LM/examples/gpt3/tokenizer/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/home/nsh/nas/projects/Megatron-LM/examples/gpt3/tokenizer/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/home/nsh/nas/projects/Megatron-LM/examples/gpt3/data/tinystories_gpt2_text_document #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 3 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --transformer-impl local        # 强制不用 transformer_engine
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 16 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 20 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1

    # === 时间日志（Megatron timers）===
    --timing-log-level 2
    --log-timers-to-tensorboard
    --log-throughput

    # === 显存日志（整体曲线）===
    --log-memory-to-tensorboard

    # === TensorBoard 基本配置 ===
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --tensorboard-log-interval 1

    # === 训练本身 ===
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --save $CHECKPOINT_PATH

    # === PyTorch Profiler（关键）===
    --profile
    --use-pytorch-profiler
    --profile-step-start 10
    --profile-step-end 13
    --profile-ranks 0

    # 可选：记录显存时间序列 & 单独的 snapshot 文件
    --record-memory-history
    --memory-snapshot-path $TENSORBOARD_LOGS_PATH/memory_trace_rank0.pickle
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
