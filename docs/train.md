# 训练指南

本文档对应当前仓库的实际启动路径：Hydra 负责组装 task，Accelerate/DeepSpeed 负责分布式，训练入口是 resVLA.scripts.train。

## 1. 环境和依赖

在目标 Python 环境中安装 [requirements.txt](../requirements.txt)：

~~~bash
python3 -m pip install -r requirements.txt
~~~

主要运行时依赖包括：

- Python、PyTorch 2.6.0、TorchVision 0.21.0；
- Transformers 5.17.0、Hydra、Accelerate、DeepSpeed；
- safetensors、datasets、pyarrow；
- av 或 torchcodec，用于 LeRobot 视频解码；
- wandb，用于外部实验跟踪。

注意：当前 requirements.txt 没有显式列出 deepspeed 和 flash-attn。ZeRO-1/2/3 配置需要环境中可导入 deepspeed；如果不使用 flash attention，则应将 model.attn_implementation 改为当前 Transformers/PyTorch 支持的 backend。

模型配置默认使用 bfloat16 和 flash_attention_2。运行前需要确认 GPU、PyTorch、CUDA 和 Transformers 的 attention backend 兼容；如果环境没有 flash-attn，应按实际环境调整 model.attn_implementation。

## 2. 准备预训练 VLM

当前 task 使用 Qwen3-VL-2B 的配置：

~~~yaml
# configs/task/resvla_libero.yaml
pretrained_model_path: /2022533089/projects/ckpt/Qwen3-VL-2B-Instruct
~~~

这个目录需要同时满足两类读取需求：

1. ResVLAProcessor.from_transformer_pretrained 从中读取 tokenizer、image processor 和 chat template；
2. MappedWeightLoader 从文件或目录中读取 safetensors 权重。

权重加载器支持一个 safetensors 文件，也支持目录下的多个 safetensors shard，会逐 shard 流式读取。当前不需要、也不提供 merge_qwen_weight.py。若 checkpoint 只有 pytorch_model.bin 而没有 safetensors，需要先准备兼容的 safetensors checkpoint，或实现新的 WeightLoader。

如果使用 Qwen3-VL-4B，应将 task 的 model 默认项改为 qwen3_vl_4b，并确保 pretrained_model_path 对应同一模型的 processor 和权重。

## 3. 准备数据和统计量

LIBERO 的数据 repo 在 [configs/dataset/transform_datasets/libero.yaml](../configs/dataset/transform_datasets/libero.yaml) 中配置，当前包含四个本地 LeRobot 路径。开始训练前需要：

1. 确认每个 repo id 可被当前环境读取；
2. 确认 lerobot_version 与数据格式匹配；
3. 确认 cam_high、cam_wrist、action、observation.state 和 task 字段存在；
4. 确认 norm_stats 下的 action/state 统计文件与 transform 链一致。

如果数据版本、动作维度或归一化前的 transform 发生变化，重新计算统计量：

~~~bash
python3 resVLA/scripts/compute_norm.py \
  --task resvla_libero \
  --batch-size 128 \
  --num-workers 8
~~~

脚本会从 task 配置中的 Normalizer 路径推导输出位置，例如：

~~~text
norm_stats/resvla_libero/libero/action_stats.json
norm_stats/resvla_libero/libero/state_stats.json
~~~

更多字段和形状说明见 [data.md](data.md)。

## 4. 启动训练

必须在仓库根目录执行。最小命令如下：

~~~bash
scripts/launch_train.sh \
  --task resvla_libero \
  --gpus 0,1,2,3
~~~

启动脚本会完成以下工作：

- 校验 task、GPU 列表和端口；
- 设置 CUDA_VISIBLE_DEVICES；
- 从 29500 开始寻找空闲的 main process port；
- 调用 accelerate launch -m resVLA.scripts.train；
- 将 task=<task_name> 传给 Hydra。

单卡调试和短跑验证可以使用：

~~~bash
scripts/launch_train.sh \
  --task resvla_libero \
  --gpus 0 \
  -- \
  num_train_steps=10 \
  log_interval=1 \
  save_interval=10 \
  pre_device_batch_size=1 \
  num_workers=0
~~~

脚本参数：

| 参数 | 说明 |
| --- | --- |
| --task | configs/task/ 下的 task 名称，不带 .yaml |
| --gpus | 逗号分隔的 GPU 编号，例如 0,1,2,3 |
| --port | 起始端口，默认 29500 |
| -- | 后续内容作为 Hydra override 原样传入 |

例如，修改训练步数或切换 ZeRO 配置：

~~~bash
scripts/launch_train.sh \
  --task resvla_libero \
  --gpus 0,1 \
  -- \
  num_train_steps=50000 \
  train/accelerator=zero3
~~~

## 5. 配置层次

resvla_libero task 组合了：

~~~text
configs/config.yaml
└── configs/task/resvla_libero.yaml
    ├── configs/train/trainer.yaml
    ├── configs/model/resVLA.yaml
    │   └── configs/model/vlm_config/qwen3_vl_2b.yaml
    ├── configs/dataset/libero_only.yaml
    │   └── configs/dataset/transform_datasets/libero.yaml
    └── configs/train/accelerator/zero2.yaml
~~~

常用可覆盖项包括：

- model.action_horizon、model.action_dim；
- pre_device_batch_size、gradient_accumulation_steps；
- num_train_steps、log_interval、save_interval；
- optimizer.lr、scheduler.num_warmup_steps；
- train/accelerator=zero1、zero2 或 zero3；
- pretrained_model_path；
- dataset.datasets[0].repo_ids（复杂结构建议直接复制 task 配置建立新 task）。

DataLoader 的 batch_size 是每张 GPU 的 micro batch；有效全局 batch size 近似为：

~~~text
pre_device_batch_size × GPU 数量 × gradient_accumulation_steps
~~~

## 6. 输出、日志和 checkpoint

Hydra 默认将结果写到：

~~~text
results/<task_name>_<MMDD>/
~~~

Trainer 每隔 save_interval 保存：

~~~text
checkpoint/step_<N>/                 # ResVLA save_pretrained 输出
optimizer/step_<N>/scheduler.pt
optimizer/step_<N>/rank_<rank>.pt    # 各进程 optimizer 状态
~~~

checkpoint/step_<N>/ 同时包含 ResVLAProcessor 的 tokenizer、image processor 和 chat template 文件，可以直接加载：

~~~python
processor = AutoProcessor.from_pretrained("results/<task>_<date>/checkpoint/step_<N>")
~~~

每隔 log_interval 记录 loss、数据读取耗时、模型耗时和每个 optimizer group 的学习率。模型返回的核心 loss 名称是：

~~~text
losses/total_loss
losses/flow_loss
losses/anchor_loss
~~~

当前 task 配置已经提供 wandb 节点，trainer 配置通过 wandb_cfg 将它传给 Trainer，Trainer 会调用 accelerator.init_trackers。由于 accelerator 配置位于 train 节点，task 级别应通过 train.accelerator.log_with 启用 W&B，否则 Accelerate 不会注册 W&B tracker。启动脚本会保留外部的 WANDB_MODE 设置；如果没有设置则默认为 online。

## 7. Accelerate + W&B 配置

推荐让 Trainer 统一管理 W&B tracker，而不是在训练脚本中直接调用 wandb.init()。可以在 task 文件中覆盖实际传给 Accelerator 的嵌套配置：

~~~yaml
train:
  accelerator:
    log_with: wandb
~~~

然后在 Trainer 的 accelerator.prepare(...) 成功之后、训练循环开始之前调用 init_trackers。当前工作区的 Trainer 已经接收 wandb_cfg 并执行初始化，但初始化位置仍在 prepare(...) 之前，建议后续调整到 prepare(...) 成功之后。W&B 参数可以集中放在 task 的 cfg.wandb 中：

~~~yaml
wandb:
  project: resvla
  name: ${task_name}_${now:%m%d}
  dir: ${hydra:run.dir}/wandb
  group: ${task_name}
~~~

初始化时把 cfg.wandb 转为普通字典，并将 project 单独传给 Accelerate：

~~~python
wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
project_name = wandb_kwargs.pop("project")

accelerator.init_trackers(
    project_name=project_name,
    init_kwargs={"wandb": wandb_kwargs},
)
~~~

config 不是必须项。如果需要在 W&B run 中保存完整实验配置，可以额外传入：

~~~python
config=OmegaConf.to_container(
    cfg,
    resolve=True,
    enum_to_str=True,
)
~~~

init_trackers 不应只在 is_main_process 分支中调用。需要上传代码时，再在主进程中获取底层 W&B run：

~~~python
if accelerator.is_main_process:
    run = accelerator.get_tracker("wandb", unwrap=True)
    run.log_code(
        ".",
        include_fn=lambda path: path.endswith((".py", ".sh")),
        exclude_fn=lambda path: "results/" in path,
    )
~~~

Trainer 中已有的 accelerator.log(metrics, step=self.step) 可以继续使用，训练结束时由 accelerator.end_training() 关闭 tracker。不要同时手动调用 wandb.init()。

## 8. 当前限制

- train.py 中 resume 分支仍是 TODO，当前不会自动恢复 step、optimizer 和 scheduler；
- Trainer 会将 BatchCollator 使用的 processor 保存到模型 checkpoint 目录；
- 仓库没有独立的评估或推理命令，模型推理需要调用 ResVLA.predict_action；
- predict_action 返回的是归一化动作，部署侧必须使用对应的 Unnormalizer；
- PadStateAction 生成的 action_mask/state_mask 会进入 batch，但当前 ResVLA.forward 尚未使用这些 mask；
- safetensors 权重必须能按 task 中的 prefix_map 映射到目标模型，否则加载时会直接报 key mismatch。

遇到数据读取失败时，先用单卡、num_workers=0、pre_device_batch_size=1 运行短跑命令，确认 repo 路径、视频 backend、processor 路径和统计文件都可用，再扩大并行度。
