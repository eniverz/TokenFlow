# ResVLA

ResVLA 先根据图像,语言,机器人状态和具身类型预测频域动作, 保留学习到的低频分量, 再通过 IDCT 得到时域 anchor；Flow Refiner 以加噪后的 anchor 为起点, 预测通向目标动作的速度场

相关实现([ResVLA](/resVLA/models/resVLA/)):

- [configuration_resvla.py](/Users/eniverz/Documents/lab/4dv/resVLA/resVLA/models/resVLA/configuration_resvla.py)：模型参数及默认值
- [modular_resvla.py](/Users/eniverz/Documents/lab/4dv/resVLA/resVLA/models/resVLA/modular_resvla.py)：DCT,频率截断,编码器,DiT,Flow Refiner 等组件
- [modeling_resvla.py](/Users/eniverz/Documents/lab/4dv/resVLA/resVLA/models/resVLA/modeling_resvla.py)：`forward`,`compute_anchor` 和 `predict_action`
- [processing_resvla.py](/Users/eniverz/Documents/lab/4dv/resVLA/resVLA/models/resVLA/processing_resvla.py) 与 [batch_collator.py](/Users/eniverz/Documents/lab/4dv/resVLA/resVLA/data/batch_collator.py)：单条样本的图文编码及 batch 拼接

详细的介绍在 [ResVLA README](/resVLA/models/resVLA/README.md) 里面

# TokenFlow

tokenflow分为三个阶段.
1. 使用tokenizer将action离散化编码为token. 使用RVQ作为核心, 使用 CNN+Perceiver 对action进行编码, 使用近似对称的decoder来还原
2. 使用离散VLA学习token, 利用第一阶段的tokenizer, 让VLM学习如何得出离散的action token
3. 在离散的action token的基础上, 还原出原来的action, 然后在此基础上进行flow matching, 得到最终的结果

## Stage 1


