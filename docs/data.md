# 数据管线

数据管线由三部分组成：

1. LeRobot reader 负责从本地或远程 repo 读取帧、状态、动作和任务文本；
2. TransformDataset 按配置顺序执行样本级 transform；
3. BatchCollator 在 DataLoader 中把图像和文本交给 ResVLAProcessor，并把数值字段拼成 batch。

当前 LIBERO 任务的入口是 [configs/dataset/transform_datasets/libero.yaml](../configs/dataset/transform_datasets/libero.yaml)，外层再由 [MixtureDataset](../resVLA/data/mixture_dataset.py) 组合。

## 数据集装配

### TransformDataset

[TransformDataset](../resVLA/data/transform_dataset.py) 接收一个或多个 LeRobot repo id。每个 repo 根据 lerobot_version 选择：

- v2.1：resVLA.data.lerobot.lerobot_dataset；
- v3.0：resVLA.data.lerobot.lerobot_dataset_v3。

构造 LeRobotDataset 时，代码会：

- 根据 metadata 的 fps 创建 action 的 delta_timestamps；
- 请求从当前帧开始的 action_horizon 个动作；
- 在 action_only=false 时读取图像和视频帧；
- 对每个读取失败的样本顺序尝试同一数据集中的下一个样本，直到找到可用样本或全部失败。

因此 action_horizon=16 时，一个样本的动作通常是 [16, action_dim_raw]，state 是当前时刻的一步状态，图像由 LeRobot reader 转换为 tensor 或 PIL 兼容对象。

### MixtureDataset

[MixtureDataset](../resVLA/data/mixture_dataset.py) 将多个 TransformDataset 组成一个 PyTorch Dataset：

- weights 缺省时每组为 1.0；
- 权重会归一化，使逻辑总长度约等于各数据集真实长度之和；
- 权重小于 1 的数据集使用确定性的跨源步长映射进行 undersampling；
- 每轮 DataLoader 重新开始时，Trainer 调用 set_epoch，改变 undersampling 的偏移；
- 当前 LIBERO 配置只有一个 TransformDataset，weights=[1.0]，所以不会改变采样分布。

这不是带随机数的 WeightedRandomSampler；它通过逻辑索引映射控制每个数据集在一个 epoch 中出现多少次。

## 样本 transform 顺序

LIBERO 的 transform 顺序非常重要，当前配置是：

~~~text
原始 LeRobot 字段
  ↓
RepackTransform
  ↓
InjectDefaultPrompt
  ↓
CoTPrompt
  ↓
InjectEmbodiment("franka")
  ↓
InjectActionTimestamp(action_horizon=16, action_freq=20)
  ↓
Normalizer(norm_type="quantile")
  ↓
PadStateAction(action_dim=32)
  ↓
ResizeImage(size=(224, 224))
~~~

### 1. RepackTransform

将 LeRobot 字段重新组织成模型侧的统一 schema：

~~~yaml
actions: action
states: observation.state
images:
  cam_high: observation.images.image
  cam_wrist: observation.images.wrist_image
prompt: task
~~~

transform 会先展开嵌套 key，再按映射读取，因此源数据必须包含这些字段，或者后续 transform 必须能够补全缺失字段。

### 2. Prompt

InjectDefaultPrompt 只在样本没有 prompt 时写入 Do task.。LIBERO 样本通常已经有 task 文本，因此默认值一般不会生效。

CoTPrompt 将现有 prompt 包装成：

~~~text
Your task is: {instruction}. Please think and generate the next action chunk based on the current observations.
~~~

prompt 仍是普通字符串，直到 BatchCollator 才会进入聊天模板。

### 3. Embodiment 和时间戳

InjectEmbodiment 写入字符串字段 embodiment=franka。这个字符串必须同时出现在模型配置的 embodiment_config 中，否则 EmbodimentEmbeddings 查表时会失败。

InjectActionTimestamp 增加 action_timestamps，值为：

~~~text
[0/20, 1/20, ..., 15/20]
~~~

该字段目前会随 batch 保留，但 ResVLA.forward 不直接使用它；它主要为下游数据处理或未来模型扩展保留。

### 4. 归一化

Normalizer 支持 min-max、quantile 和 mean-std 三种方式。LIBERO 使用 quantile，把统计量 q01 和 q99 映射到 [-1, 1]：

~~~text
x_normalized = 2 * (x - q01) / (q99 - q01 + 1e-8) - 1
~~~

统计量来自：

- [action_stats.json](../norm_stats/resvla_libero/libero/action_stats.json)
- [state_stats.json](../norm_stats/resvla_libero/libero/state_stats.json)

统计文件的 mask 决定哪些维度参与归一化。当前实现不会自动 clip 超出 q01/q99 的值。

### 5. Padding 和 mask

PadStateAction 将动作和状态的最后一维补到 action_dim=32，补零并生成 mask：

~~~text
actions      [16, action_dim_raw] → [16, 32]
states       [1, state_dim_raw]   → [1, 32]
action_mask  [32]                 # 原始动作维度为 True
state_mask   [32]                 # 原始状态维度为 True
~~~

mask 会进入 batch，但当前 ResVLA.forward 只读取 actions 和 states，尚未用 mask 对损失进行逐维屏蔽。若新增 embodiment 的有效维度不同，应同时确认模型损失是否需要改为 mask-aware。

### 6. 图像

ResizeImage 将 tensor 或 NumPy 数组转换为 RGB PIL.Image，保持纵横比缩放后居中 padding 到 224×224。LIBERO 的两路相机顺序来自 images 字典的插入顺序，当前为 cam_high、cam_wrist。

## BatchCollator 的输出

[BatchCollator](../resVLA/data/batch_collator.py) 对每个样本执行 ResVLAProcessor.encode_input：

1. 按 images.values() 的顺序加入图像；
2. 追加 prompt 文本；
3. 以一个 user message 调用 Qwen3VLProcessor.apply_chat_template；
4. 对 input_ids、attention_mask 等序列字段做 padding；
5. 对视觉 patch/grid 字段按 batch 维拼接；
6. 对其余字段使用 PyTorch default_collate。

传给模型的 batch 形状可以理解为：

~~~text
{
  "actions":       Tensor[B, 16, 32],
  "states":        Tensor[B, 1, 32],
  "action_mask":   Tensor[B, 32],
  "state_mask":    Tensor[B, 32],
  "embodiment":    List[str]              # 长度 B
  "action_timestamps": Tensor[B, 16],
  "vlm_inputs": {
      "input_ids": ...,
      "attention_mask": ...,
      # 以及 Qwen 视觉输入字段，如 pixel_values、image_grid_thw
  }
}
~~~

sample 中不能预先包含 vlm_inputs；这是 BatchCollator 保留的字段名。图像不会直接 stack 成 tensor，而是先保留为 PIL 图像列表，交给 processor 处理。

## 计算统计量

[resVLA/scripts/compute_norm.py](../resVLA/scripts/compute_norm.py) 会读取 task 配置，在第一个 Normalizer 之前执行 transform，然后单独为每个 TransformDataset 统计 action/state。典型命令：

~~~bash
python3 resVLA/scripts/compute_norm.py \
  --task resvla_libero \
  --batch-size 128 \
  --num-workers 8
~~~

脚本会按照 Normalizer 中配置的路径写回 norm_stats/<task>/<dataset>/。MixtureDataset 的权重不会参与统计；一个 TransformDataset 中列出的所有 repo 会共同贡献统计量。

如果数据字段、action horizon、delta/absolute action 变换或数据版本发生变化，应先重新计算统计量，再启动训练。训练配置中的相对路径以仓库根目录为基准。

## 添加新数据集

建议按以下顺序接入：

1. 在 configs/dataset/transform_datasets/ 新增 YAML，明确 repo_ids、lerobot_version、action_horizon、action_only 和 video_backend。
2. 用 RepackTransform 将源字段映射为 actions、states、images、prompt。
3. 注入与模型配置一致的 embodiment 字符串。
4. 选择 Normalizer，并为该 TransformDataset 生成独立的 action/state 统计文件。
5. 根据原始动作维度设置 action_dim；如果不同数据集需要不同维度，应考虑拆分模型配置或在 transform 中统一 padding。
6. 用一个小 batch 检查 actions、states、images、prompt、embodiment 的类型和形状，再运行完整训练。

如果使用 v3 数据集，确保 lerobot_version 写为 v3.0；TransformDataset 当前只接受 v2.1 和 v3.0 两种值。
