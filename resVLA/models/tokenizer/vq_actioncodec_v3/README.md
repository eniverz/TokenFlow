v3 在 v2 基础上增加规则夹爪编解码，训练配置为 `task=tokenflow_stage1_v3`。

- `modular_vq_codec.py` 中的 `RuleBasedCodec` 使用二值化、短中间段合并、字典序排名和 K 进制转换。
- 当前 LIBERO 夹爪保持原始 0/1 数值，二值化阈值为 0.5，解码输出 0.0/1.0。
- `action_horizon=16`、`codebook_size=4096`、`rule_based_min_block_len=1` 时，每个夹爪用一个 token。
- sample 格式不变；`forward(samples)` 仅训练 NN 部件，夹爪不参与 augmentation 或任何训练 loss。
- 规则编解码按单条序列调用，不提供模型级统一 encode/decode 接口。

对 collate 后的 batch，可直接使用现有 mask 提取夹爪：

```python
name = "right_gripper"
present = samples["actions_mask"][name].any(dim=-1)
sequences = samples["actions_dict"][name][present, :, 0].tolist()
codes = [model.rule_codec.encode(sequence) for sequence in sequences]
reconstructed = [model.rule_codec.decode(tokens) for tokens in codes]
```

中间连续段长度必须大于 `rule_based_min_block_len`，首尾段不受限制。
编码会从左到右合并过短的中间段，因此解码恢复的是修复后的二值轨迹。
规则路径没有参数校验、越界截断、fallback 或异常捕获。
