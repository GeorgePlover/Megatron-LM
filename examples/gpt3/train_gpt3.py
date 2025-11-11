#!/usr/bin/env python
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional
CHECK_RUN_CMD = True # 是否检查运行命令，如果运行失败将会中断程序

# ================= 基本配置结构体 =================

@dataclass
class GPTModelConfig:
    num_layers: int = 3
    hidden_size: int = 512
    num_attention_heads: int = 8
    seq_length: int = 1024
    max_position_embeddings: int = 1024
    attention_backend: str = "auto"   # auto / flash / fused / unfused / local
    transformer_impl: str = "local"   # 强制不用 transformer_engine


@dataclass
class TrainConfig:
    micro_batch_size: int = 1
    global_batch_size: int = 16
    train_iters: int = 20

    # 下面这些一般不怎么改，保留为可配置
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    init_method_std: float = 0.006
    clip_grad: float = 1.0
    fp16: bool = True
    lr: float = 6.0e-5
    lr_decay_style: str = "cosine"
    min_lr: float = 6.0e-6
    lr_warmup_fraction: float = 0.001
    lr_decay_iters: int = 430000


@dataclass
class ProfileConfig:
    enable_profile: bool = True
    step_start: int = 10
    step_end: int = 13
    profile_ranks: str = "0"   # Megatron 接受类似 "0" 或 "0,1"


# ================= 命令构建函数 =================

def build_megatron_command(
    log_dir: str,
    tensor_parallel_size: int,
    gpt_model_config: GPTModelConfig,
    train_config: TrainConfig,
    profile_config: ProfileConfig,
    data_path: str,
    vocab_file: str,
    merge_file: str,
    gpus_per_node: int = 2,
    num_nodes: int = 1,
    master_addr: str = "localhost",
    master_port: int = 6000,
    pipeline_model_parallel_size: int = 1,
    pretrain_script: str = "pretrain_gpt.py",
) -> List[str]:
    """
    构建用于运行 Megatron-LM pretrain_gpt.py 的 torchrun 命令。
    """

    # 基本检查
    world_size = gpus_per_node * num_nodes
    if tensor_parallel_size > world_size:
        raise ValueError(
            f"TP={tensor_parallel_size} 大于总进程数 {world_size}，请调小 TP 或增大 GPU 数。"
        )

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    memory_snapshot_path = os.path.join(log_dir, "memory_trace_rank0.pickle")

    # 分布式 / torchrun 参数
    cmd = [
        "torchrun",
        "--nproc_per_node", str(gpus_per_node),
        "--nnodes", str(num_nodes),
        "--master_addr", master_addr,
        "--master_port", str(master_port),
        pretrain_script,
    ]

    # ===== GPT 模型参数 =====
    cmd += [
        "--num-layers", str(gpt_model_config.num_layers),
        "--hidden-size", str(gpt_model_config.hidden_size),
        "--num-attention-heads", str(gpt_model_config.num_attention_heads),
        "--seq-length", str(gpt_model_config.seq_length),
        "--max-position-embeddings", str(gpt_model_config.max_position_embeddings),
        "--attention-backend", gpt_model_config.attention_backend,
        "--transformer-impl", gpt_model_config.transformer_impl,
    ]

    # ===== 训练参数 =====
    cmd += [
        "--micro-batch-size", str(train_config.micro_batch_size),
        "--global-batch-size", str(train_config.global_batch_size),
        "--train-iters", str(train_config.train_iters),
        "--weight-decay", str(train_config.weight_decay),
        "--adam-beta1", str(train_config.adam_beta1),
        "--adam-beta2", str(train_config.adam_beta2),
        "--init-method-std", str(train_config.init_method_std),
        "--clip-grad", str(train_config.clip_grad),
        "--lr", str(train_config.lr),
        "--lr-decay-style", train_config.lr_decay_style,
        "--min-lr", str(train_config.min_lr),
        "--lr-warmup-fraction", str(train_config.lr_warmup_fraction),
        "--lr-decay-iters", str(train_config.lr_decay_iters),
    ]
    if train_config.fp16:
        cmd.append("--fp16")

    # ===== 模型并行参数（含 TP）=====
    cmd += [
        "--tensor-model-parallel-size", str(tensor_parallel_size),
        "--pipeline-model-parallel-size", str(pipeline_model_parallel_size),
    ]

    # ===== 数据参数 =====
    cmd += [
        "--data-path", data_path,
        "--vocab-file", vocab_file,
        "--merge-file", merge_file,
        "--split", "949,50,1",
    ]

    # ===== 日志 / TensorBoard / Memory Log =====
    cmd += [
        "--log-interval", "1",
        "--timing-log-level", "2",
        "--log-timers-to-tensorboard",
        "--log-throughput",
        "--log-memory-to-tensorboard",
        "--tensorboard-dir", tensorboard_dir,
        "--tensorboard-log-interval", "1",
        "--eval-interval", "1000",
        "--eval-iters", "10",
        "--record-memory-history",
        "--memory-snapshot-path", memory_snapshot_path,
    ]

    # ===== Profiling（可控区间）=====
    if profile_config.enable_profile:
        cmd += [
            "--profile",
            "--use-pytorch-profiler",
            "--profile-step-start", str(profile_config.step_start),
            "--profile-step-end", str(profile_config.step_end),
            "--profile-ranks", profile_config.profile_ranks,
        ]

    # ===== 重要：不包含任何 checkpoint 保存相关参数 =====
    # 不加 --save, --save-interval 等，避免 IO 开销。

    return cmd


# ================= 对外调用接口 =================

def run_megatron_gpt(
    log_dir: str,
    tensor_parallel_size: int = 1,
    gpt_model_config: Optional[GPTModelConfig] = None,
    train_config: Optional[TrainConfig] = None,
    profile_config: Optional[ProfileConfig] = None,
    data_path: str = "/home/nsh/nas/projects/Megatron-LM/examples/gpt3/data/tinystories_gpt2_text_document",
    vocab_file: str = "/home/nsh/nas/projects/Megatron-LM/examples/gpt3/tokenizer/gpt2-vocab.json",
    merge_file: str = "/home/nsh/nas/projects/Megatron-LM/examples/gpt3/tokenizer/gpt2-merges.txt",
    gpus_per_node: int = 2,
    num_nodes: int = 1,
    master_addr: str = "localhost",
    master_port: int = 6000,
    pipeline_model_parallel_size: int = 1,
    pretrain_script: str = "pretrain_gpt.py",
    dry_run: bool = False,
):
    """
    一行调用入口：
    - log_dir: 本次实验所有日志/Profiler输出目录（你每次实验只需要改这个）
    - tensor_parallel_size: TP 度数，一般用 1 或 2
    - gpt_model_config: GPTModelConfig 实例（可选）
    - train_config: TrainConfig 实例（可选）
    - profile_config: ProfileConfig 实例（可选）
    - dry_run: True 时只打印命令，不实际运行
    """

    if gpt_model_config is None:
        gpt_model_config = GPTModelConfig()
    if train_config is None:
        train_config = TrainConfig()
    if profile_config is None:
        profile_config = ProfileConfig()

    cmd = build_megatron_command(
        log_dir=log_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpt_model_config=gpt_model_config,
        train_config=train_config,
        profile_config=profile_config,
        data_path=data_path,
        vocab_file=vocab_file,
        merge_file=merge_file,
        gpus_per_node=gpus_per_node,
        num_nodes=num_nodes,
        master_addr=master_addr,
        master_port=master_port,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pretrain_script=pretrain_script,
    )

    print("[Megatron GPT] Launch command:")
    print(" ".join(cmd))

    if not dry_run:
        env = os.environ.copy()
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        subprocess.run(cmd, check=CHECK_RUN_CMD, env=env)

GPTModelConfigs = {
    "117m": GPTModelConfig(
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        seq_length=1024,
        max_position_embeddings=1024,
    ),
    "345m": GPTModelConfig(
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        seq_length=1024,
        max_position_embeddings=1024,
    ),
    "760m": GPTModelConfig(
        num_layers=24,
        hidden_size=1536,
        num_attention_heads=16,
        seq_length=1024,
        max_position_embeddings=1024,
    ),
    "1.3b": GPTModelConfig(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        seq_length=2048,
        max_position_embeddings=2048,
    ),
    "6.7b": GPTModelConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        seq_length=2048,
        max_position_embeddings=2048,
    ),
    "13b": GPTModelConfig(
        num_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
        seq_length=2048,
        max_position_embeddings=2048,
    ),
    # "175b": GPTModelConfig( # 模型太大了，放不下
    #     num_layers=96,
    #     hidden_size=12288,
    #     num_attention_heads=96,
    #     seq_length=2048,
    #     max_position_embeddings=2048,
    # ),
}

# ================= 使用示例 =================
if __name__ == "__main__":
    # 示例：TP=2，小模型，输出到指定目录，不保存 checkpoint
    
    for model_name, gpt_cfg in GPTModelConfigs.items():
        for layernum in range(1, 5):
            for tp_size in [1, 2]:
                print(f"Training {model_name} with {layernum} layers")
                folder_name = f"gpt3_{model_name}_{layernum}layers_tp{tp_size}"
                log_dir = "/home/nsh/nas/projects/Megatron-LM/profile_runs/" + folder_name
                if os.path.exists(log_dir):
                    print(f"Folder {log_dir} already exists, skipping.")
                    continue
                
                gpt_cfg.num_layers = layernum
                
                train_cfg = TrainConfig(
                    micro_batch_size=1,
                    global_batch_size=8,
                    train_iters=50,
                )

                profile_cfg = ProfileConfig(
                    enable_profile=True,
                    step_start=40,
                    step_end=42,
                    profile_ranks="0",
                )

                run_megatron_gpt(
                    log_dir=log_dir,
                    tensor_parallel_size=tp_size,
                    gpt_model_config=gpt_cfg,
                    train_config=train_cfg,
                    profile_config=profile_cfg,
                    gpus_per_node=tp_size,          # 单机两卡做 TP=2
                    num_nodes=1,
                    dry_run=False,            # 改成 True 可以只看命令不执行
                )
