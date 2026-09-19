# ResVLA 架构

本文档描述当前代码的实际组织方式和一次训练迭代中的数据流。仓库使用 Hydra 组装配置，使用 Hugging Face Transformers 的 PretrainedConfig/PreTrainedModel 组织模型，使用 Accelerate + DeepSpeed 执行多卡训练。

## 代码分层

~~~text
configs/
├── config.yaml                         # Hydra 根配置、结果目录和日志
├── task/                               # 一次实验的组合入口
├── model/                              # ResVLA 与 VLM 配置
├── dataset/                            # 数据集和 transform 链
└── train/                              # Trainer、优化器和 ZeRO 配置
resVLA/
├── models/resVLA/                      # ResVLAConfig、ResVLA、ResVLAProcessor
├── models/tokenizer/vq_codec/          # VQ codec 预留目录，当前未接入训练主路径
├── data/                               # LeRobot 读取、混合数据集、collator、transforms
├── training/                           # Trainer 和 safetensors 权重加载器
├── optim/                              # 分组 AdamW 和学习率调度器
└── scripts/train.py                    # Hydra 训练入口
scripts/launch_train.sh                 # 多卡启动包装脚本
norm_stats/                             # Normalizer 使用的 action/state 统计量
~~~

各层之间的依赖方向是：

~~~mermaid
flowchart LR
    C[Hydra task config] --> D[Dataset + transforms]
    C --> M[ResVLAConfig]
    M --> N[ResVLA]
    D --> B[BatchCollator]
    B --> P[ResVLAProcessor]
    P --> V[vlm_inputs]
    V --> N
    N --> R[ResVLA.forward]
    R --> T[Trainer]
    T --> O[Optimizer + Scheduler]
    T --> S[checkpoint / W&B]
~~~

模型、数据集和 Trainer 都通过配置中的 _target_ 由 hydra.utils.instantiate 创建。新增组件时，通常只需要提供一个可实例化的 Python 类并在 YAML 中引用它。

## 模型注册和初始化

训练进程启动时，resVLA.scripts.train 会先调用 resVLA.models.register_models.register_all()，将以下类型注册到 Transformers 的 Auto 类：

- model_type: resvla → [ResVLAConfig](../resVLA/models/resVLA/configuration_resvla.py)；
- ResVLAConfig → [ResVLA](../resVLA/models/resVLA/modeling_resvla.py)；
- ResVLAConfig → [ResVLAProcessor](../resVLA/models/resVLA/processing_resvla.py)。

随后 Hydra 按以下顺序创建模型：

1. 实例化 cfg.model，得到 ResVLAConfig。
2. 执行 AutoModel.from_config(model_cfg)，构造 ResVLA 和内部 VLM backbone。
3. 实例化 cfg.weight_loader，把预训练 VLM 权重加载到 model.vlm。
4. 按配置创建优化器、调度器和 Trainer。

ResVLAConfig.vlm_config 是一个嵌套的 Transformers 配置，默认由 configs/model/vlm_config/qwen3_vl_2b.yaml 提供。模型的 hidden_size 直接取 VLM 的 text_config.hidden_size，所以切换 Qwen 2B/4B 时，ResVLA 的隐藏维度会随 VLM 配置变化。

当前默认模型关键参数如下：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| action_horizon | 16 | 每个样本预测的动作步数 |
| action_dim | 32 | padding 后的动作/状态特征维度 |
| num_anchor_layers | 19 | Anchor DiT 层数 |
| num_refiner_layers | 9 | Flow Refiner 层数 |
| num_inference_steps | 4 | 推理时 Euler 积分步数 |
| num_repeat_diffusion_steps | 2 | 训练时每个样本重复的 flow matching 轨迹数 |
| embodiment_config | ["franka"] | 支持的 embodiment 名称列表 |

### 预训练权重加载

[MappedWeightLoader](../resVLA/training/weight_loader.py) 支持传入一个 safetensors 文件或包含多个 *.safetensors shard 的目录。resvla_libero 的配置使用：

~~~yaml
prefix_map:
  "model.": "vlm."
exclude_keys:
  - lm_head.weight
~~~

因此 Qwen checkpoint 中的 model.xxx 会映射到 ResVLA 中的 vlm.xxx，语言模型输出头暂不加载。加载器会逐 shard 流式读取，并检查每个映射后的目标 key 是否存在；不需要额外的权重合并脚本。

## ResVLA 的前向计算

ResVLA.forward 接收一个 batch 字典，核心字段是：

~~~text
vlm_inputs                 Qwen processor 产生的输入字典
states       [B, 1, 32]    当前状态（已归一化并 padding）
actions      [B, 16, 32]   目标动作 chunk（已归一化并 padding）
embodiment   List[str]     长度为 B 的 embodiment 名称
~~~

一次训练前向可概括为：

1. VLM 以 bfloat16 前向，并保留所有 hidden states。每个 VLM hidden state 都可以作为后续 cross-attention 的上下文。
2. EmbodimentEmbeddings 将 embodiment 字符串查表，再由 EmbodimentEncoder 得到条件向量；ActionStateEncoder 将当前状态编码到 VLM 隐藏维度。
3. **Anchor DiT** 使用状态 token、VLM hidden states、可学习的 future tokens 和 cutoff token，生成粗粒度的动作表示。
4. AnchorMLP 将 future token 解码到 DCT 频域。LearnableCutoffProjector 对 cutoff token 产生硬的频率 mask，只保留从低频开始的一段分量；anchor loss 与真实动作的 DCT 结果比较。
5. 频域 anchor 经过固定的 IDCT 回到时域，作为 flow matching 的初始轨迹 x_0。真实动作是 x_1，中间状态按采样的时间 t 线性插值，并加入 fm_noise_std 控制的噪声。
6. **Flow Refiner** 以时间编码、embodiment 条件、状态 token 和最后 num_refiner_layers 个 VLM hidden states 为条件，预测速度场。
7. 总损失为 flow loss 加上按比例缩放的 anchor loss：

~~~text
total_loss = flow_loss
           + (flow_loss.detach() / (anchor_loss.detach() + 1e-8))
             * anchor_loss_weight * anchor_loss
~~~

返回值是一个字典，loss 是 Trainer 必须反向传播的总损失；同时返回 losses/total_loss、losses/flow_loss 和 losses/anchor_loss 供日志记录。

predict_action 使用同一个 anchor 阶段，然后从带噪声的 anchor 开始进行 num_inference_steps 次 Euler 积分，返回形状为 [B, action_horizon, action_dim] 的**归一化动作**。当前仓库没有单独的推理脚本，部署侧需要自行调用该方法并使用与训练相同的 Unnormalizer 还原动作尺度。

## 优化和训练循环

[GroupedAdamW](../resVLA/optim/optimizer.py) 按 module path 建立参数组。LIBERO 配置为 VLM language model、VLM visual、anchor_dit 和 flow 设置不同学习率，未被显式分组的参数进入 default 组。

[WarmupConstantCosineLR](../resVLA/optim/scheduler.py) 对每个参数组使用相同的倍率：先线性 warmup，随后保持一段时间，最后做 cosine decay；各组之间的学习率比例保持不变。

[Trainer](../resVLA/training/trainer.py) 的每个优化步执行：

1. 从 DataLoader 取一个 micro batch；
2. 在 accelerator.accumulate 中调用 model(batch) 并反向传播；
3. 更新 optimizer；只有真正同步梯度的步才更新 scheduler 和全局 step；
4. 在 log_interval 处聚合 loss、数据耗时、模型耗时和各参数组学习率；
5. 在 save_interval 处保存 checkpoint。

### Accelerate + W&B 日志

Trainer 已经通过 accelerator.log(metrics, step=self.step) 记录训练指标，但要真正创建 W&B run，还需要在 Accelerator 配置中启用 tracker：

~~~yaml
log_with: wandb
~~~

推荐由 Trainer 统一管理 tracker 生命周期：在 accelerator.prepare(...) 成功之后、训练循环开始之前调用 accelerator.init_trackers(...)。这样数据、模型或 optimizer 准备失败时不会留下空的 W&B run。

当前工作区已经在 task 配置中提供 wandb 节点，并通过 trainer.wandb_cfg 传入 Trainer；Trainer 也已经调用 init_trackers。由于 accelerator 配置位于 train 节点，task 级别的覆盖应写在 train.accelerator.log_with。当前实现是在 prepare(...) 之前初始化 run，后续可以将初始化移动到 prepare(...) 成功之后，以避免准备阶段失败时创建空 run。

task 配置可以把 W&B 参数集中在 cfg.wandb：

~~~yaml
wandb:
  project: resvla
  name: ${task_name}_${now:%m%d}
  dir: ${hydra:run.dir}/wandb
  group: vla-train
~~~

初始化时将 cfg.wandb 转换为普通字典，把 project 单独传给 project_name：

~~~python
wandb_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
project_name = wandb_kwargs.pop("project")

accelerator.init_trackers(
    project_name=project_name,
    init_kwargs={"wandb": wandb_kwargs},
)
~~~

config 参数不是必需的。如果希望在 W&B 中保存完整的 Hydra 配置，可以额外传入 OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)；不传时，指标记录和 W&B run 仍然可以正常工作。

init_trackers 不应只在主进程分支中调用，应让 Accelerate 管理分布式 tracker。只有访问底层 W&B run 并调用 log_code 时才需要主进程保护：

~~~python
if accelerator.is_main_process:
    run = accelerator.get_tracker("wandb", unwrap=True)
    run.log_code(
        ".",
        include_fn=lambda path: path.endswith((".py", ".sh")),
        exclude_fn=lambda path: "results/" in path,
    )
~~~

训练结束时继续调用 accelerator.end_training()，由 Accelerate 关闭 tracker。不要在同一进程中再直接调用 wandb.init()，否则多卡训练可能产生重复 run。

结果目录由 configs/config.yaml 设为 results/<task_name>_<MMDD>/。每个 checkpoint 的结构是：

~~~text
results/<task>_<date>/
├── checkpoint/step_<N>/                 # save_pretrained 的模型和 config
└── optimizer/step_<N>/
    ├── scheduler.pt
    └── rank_<process_index>.pt           # 各进程的 optimizer 状态
~~~

Trainer 会把 DataLoader 的 BatchCollator.processor 保存到同一个 checkpoint/step_<N>/ 目录，因此可以直接使用 AutoProcessor.from_pretrained(checkpoint_dir) 恢复 processor。当前 resume 参数仍是 TODO；task 需要通过 train.accelerator.log_with 启用 W&B，否则 init_trackers 不会注册 W&B tracker。恢复训练时仍需要显式管理 optimizer、scheduler 和 step 状态。

## 扩展约定

- 新 VLM：提供 Transformers 配置，并在 configs/model/vlm_config/ 增加 YAML；确保其 text_config.hidden_size 与内部模块可用。
- 新 embodiment：在模型配置的 embodiment_config 中加入名称，并在数据 transform 中注入完全相同的字符串。
- 新数据集：实现或复用 TransformDataset 的输入字段，保证 transform 最终产生 actions、states、images、prompt 和 embodiment。
- 新模型：继承 PreTrainedModel/PretrainedConfig，在 register_models.py 注册，再通过 task YAML 接入训练流程。

models/tokenizer/vq_codec/ 目前只有配置和模型骨架文件，训练入口没有实例化它；接入前需要补齐 modeling 实现、Auto 类注册和对应的 task 配置。
