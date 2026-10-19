# DataLoader

当前数据入口支持 LeRobot Dataset v2.1 和 v3.0，需要在 YAML 中明确指定 `lerobot_version: v2.1` 或 `lerobot_version: v3.0`。实际分支判断使用这两个完整字符串。

## Dataset

数据集按以下层级组织：

1. YAML 中的 `repo_ids` 指定一个或多个 LeRobot 数据集，`lerobot_version` 决定使用 [LeRobot v2.1](/resVLA/data/lerobot/lerobot_dataset.py) 或 [LeRobot v3.0](/resVLA/data/lerobot/lerobot_dataset_v3.py)。同一个 `TransformDataset` 中的 `repo_ids` 共用版本和 transform 配置。
2. 配置中的 `transform` 列表构成顺序变换链，在每次读取单条 sample 后依次执行。
3. 多个 LeRobot 子数据集与同一条变换链组成一个 [TransformDataset](/resVLA/data/transform_dataset.py)，其长度是所有子集长度之和。
4. 多个 `TransformDataset` 及对应的权重组成一个 [MixtureDataset](/resVLA/data/mixture_dataset.py)。每个 `TransformDataset` 是一个采样组；权重作用于整组，不分别作用于组内的每个 LeRobot 子集。
5. DataLoader 的 sampler 决定访问哪些 mixture 全局索引及其顺序，`collate_fn` 将取出的多条 sample 合成 batch。

### TransformDataset

实现见 [transform_dataset.py](/resVLA/data/transform_dataset.py)。

初始化时创建各个 LeRobot 子集，记录 `all_lengths` 及总长度 `length`，形成逻辑上的连续索引空间。这里没有把所有 sample 预先读取并展平成内存列表。

创建 LeRobot 子集时，还会根据其 metadata 中的 `fps` 设置：

```python
delta_timestamps = {
    "action": [t / fps for t in range(action_horizon)]
}
```

这样每次取样可得到从当前时刻开始的 `action_horizon` 步动作。`action_only=True` 时设置 `load_images=False`；该选项控制图像加载，并不表示 sample 必然只含动作字段。`action_horizon` 虽被注解为 float，实际用于 `range`，应配置为正整数。

数据采样流程：

1. `__getitem__(index)` 从第一个子集开始，逐次减去子集长度，定位对应子集与局部索引。例如子集长度为 `[100,300]`，组内索引 120 对应第二个子集的索引 20。
2. 调用该 LeRobot 子集的 `__getitem__`，读取一条 sample。
3. 将 sample 依次传给配置中的 transform，并返回最后的结果。
4. 如果读取或任何一个 transform 抛出异常，则在**同一个 LeRobot 子集**内尝试下一个索引，到末尾后取模绕回；最多尝试该子集长度次。全部失败后抛出 `RuntimeError`，不会自动切换到其他子集。

因此发生读取失败时，实际返回的 sample 可能不是最初索引指定的那条。Mixture 的采样数量和去重性质描述的是请求索引，不能保证异常回退后的实际样本仍无重复。

### MixtureDataset

实现见 [mixture_dataset.py](/resVLA/data/mixture_dataset.py)。该类先用权重为每组建立逻辑长度，再将全局索引确定性地映射到某组的真实索引；它不在每次 `__getitem__` 时随机抽取一个数据集，也不预先生成完整的采样顺序列表。

数据采样流程：

1. 根据各组真实长度与输入权重，计算归一化权重、逻辑长度及逻辑索引区间。
2. 根据传入的 mixture 全局索引，定位采样组及组内逻辑索引。
3. 将组内逻辑索引映射为该 `TransformDataset` 的真实索引：过采样时循环重复，欠采样时使用互质步长与 epoch 偏移。
4. 调用 `self.datasets[dataset_idx][local_idx]`，由 `TransformDataset` 完成真实样本读取与变换，再原样返回 sample。

采样数据计算流程

1. **计算每组的采样次数。** 将各组的“真实长度 × 权重”按比例缩放，使总量接近原始总长度，再取整且每组至少保留 1 次，得到逻辑长度 `effective_lengths`。因此采样占比由长度和权重共同决定，相同权重不代表各组等概率。
2. **划分索引区间。** 按逻辑长度依次排列各组，形成连续的全局索引空间。传入 `index` 后，先找到所属组，再得到组内索引。
3. **映射到真实样本。** 逻辑长度大于或等于真实长度时，通过取模循环取样；逻辑长度较小时，用与真实长度互质的步长选取不重复的索引，并根据 `set_epoch` 设置的 epoch 调整偏移。
4. **读取并返回。** 用映射后的真实索引调用对应的 `TransformDataset`，读取 sample、执行变换并返回。

例如，两组长度为 `[100,300]`、权重为 `[3,1]`，最终逻辑长度为 `[200,200]`：第一组的 100 条数据各取两次，第二组从 300 条中选取 200 条。权重默认全部为 1，显式设置时必须大于 0；取整可能使最终总长度略有变化。

#### 与 DataLoader 的分工

Mixture 自身按组构造连续逻辑区间。若外部按 `0,1,2,…` 访问，会先取完第一组再取第二组；shuffle 发生在 DataLoader/sampler 层，而非 mixture 内部。当前 [trainer.yaml](/configs/train/trainer.yaml) 设置了 `shuffle: true` 和 `drop_last: true`。

[Trainer](/resVLA/training/trainer.py) 在迭代耗尽后调用 `dataset.set_epoch(self.epoch)` 再创建迭代器。需要区分“主进程调用了 set_epoch”和“执行取样的对象已收到新 epoch”：`MixtureDataset` 使用普通整数保存 epoch，没有共享状态或 worker 同步机制；当前配置还开启了 `persistent_workers: true`，因此不能仅凭这个调用就认为持久 worker 中的欠采样偏移已经更新。这里记录的是代码的接口行为，实际跨进程传播需要单独保证。

## Transform

transform 按 YAML 列表顺序执行，后一个 transform 接收前一个 transform 返回的数据。常用项包括：

下表描述单条 sample 的输出，不含 batch 维；`H` 为动作步数，`A` 为补齐后的 `action_dim`，`M` 为 `max_part_dim`。各变换均返回数据字典，输出列列出其中新增或修改的字段。

| 实现 | 作用 | 输出结果 |
| --- | --- | --- |
| [RepackTransform](/resVLA/data/transform/process.py) | 按映射组织字段；映射的 key 为输出字段，value 为源字段，缺失源字段会得到 `None` | 仅包含映射目标字段的新字典，例如 `{"actions": ..., "states": ..., "images": {...}, "prompt": ...}`；字段值的类型和 shape 不变 |
| [InjectDefaultPrompt / CoTPrompt](/resVLA/data/transform/prompt.py) | 前者仅在没有 `prompt` 键时补默认文本；后者按模板改写已有 `prompt` | `prompt`：默认文本字符串（默认 `"Do task."`）或按模板生成的指令字符串 |
| [InjectEmbodiment](/resVLA/data/transform/embodiment.py) | 写入具身名称字符串 | `embodiment`：字符串，例如 `"franka"` |
| [Normalizer / StepwiseNormalizer](/resVLA/data/transform/action.py) | 根据统计量和有效维度归一化；后者使用逐时间步统计量 | `actions`：归一化后的 tensor，shape 不变；`states`：归一化后的 tensor，一维输入 `[S]` 会扩展为 `[1,S]`，其中 `S` 为当前状态维度 |
| [PadStateAction](/resVLA/data/transform/action.py) | 补齐动作/状态的最后一维，生成 `action_mask`、`state_mask` | `actions [H,A]`、`action_mask [A]`；`states` 最后一维补齐到 `A`（如 `[1,A]`）、`state_mask [A]`；mask 为 bool，补齐位置为 False |
| [PadActionToDict](/resVLA/data/transform/action.py) | 按 `dim_config` 提取部件动作并补齐，生成 `actions_dict`、`actions_mask`；原 `actions` 字段仍保留 | `actions_dict[part_name]`：tensor `[H,M]`；`actions_mask[part_name]`：bool tensor `[M]`，表示该部件的有效动作维度 |
| [InjectActionTimestamp](/resVLA/data/transform/action.py) | 生成 `arange(action_horizon)/action_freq`；它不负责从数据集中读取动作帧 | `action_timestamps`：tensor `[H]`，值为 `[0, 1/f, ..., (H-1)/f]`，其中 `f=action_freq` |
| [ResizeImage](/resVLA/data/transform/image.py) | 转为 RGB PIL 图像，保持比例缩放并居中补边到目标尺寸 | `images`：相机名到 RGB PIL 图像的字典，每张图的 `.size` 为配置的 `(width,height)` |

具体采用哪些变换由任务配置决定，并非每条数据都执行全部 transform。动作相关输出以输入包含非空 `actions` 为前提；缺少动作时，Normalizer 和 PadStateAction 仍处理状态，PadActionToDict 原样返回数据。

## Collate Function

DataLoader 的参数名是 `collate_fn`。图文编码实现是 [BatchCollator](/resVLA/data/batch_collator.py)，不是名为 `collator_fn` 的函数。

### 图文任务：BatchCollator

`BatchCollator.__call__` 接收多条 sample 组成的列表，主要执行：

1. 对每条 sample 的 `images`、`prompt` 调用 processor 的 `encode_input`；如果 processor 没有该方法，则通过默认图文消息模板调用 `apply_chat_template`。
2. 对原始 samples 使用 `default_collate`，堆叠动作、状态和 mask 等字段；PIL 图像通过本文件注册的规则保留为列表。
3. 对编码后的 `input_ids`、`attention_mask` 等序列字段进行 padding：单样本 `[1,Sᵢ]` 变为 `[B,S_max]`。padding 方向来自显式 `padding_side` 或 tokenizer 配置；ID 使用 pad token，attention mask 补 0。
4. 对图像 patch、grid 等其他编码字段沿第 0 维 `cat`，将结果放入 `batch["vlm_inputs"]`。图像字段按 processor 的布局打包，不一定具有独立的 batch 维。

因此，若单条动作 shape 为 `[H,A]`，batch 中为 `[B,H,A]`；单条状态为 `[1,A]` 时，batch 中为 `[B,1,A]`。原始字段仍保留，`vlm_inputs` 是新增的保留字段，sample 不应预先提供它。

### 纯动作任务：default_collate

并非所有任务都需要图文 processor。例如 [tokenflow_stage1_v3.yaml](/configs/task/tokenflow_stage1_v3.yaml) 显式将 `collate_fn` 替换为 PyTorch `default_collate`，直接递归堆叠动作字典与 mask，不生成 `vlm_inputs`。选择哪个 collator 应以具体任务配置为准。
