# Model Design

设计理念:
- 所有的模型使用 PreTrainedModel 管理, 使用 PretrainedConfig 作为配置类
- 一个模型的所有文件都要放在同一个文件夹里面.
    - configuration_xxx.py 用于记录所有的config配置
    - modular_xxx.py 用于存放所有的组件, 里面放各种需要用到的 nn.Module. 注意, 这里面只能使用 torch/einops/numpy 等 和基础的 python builtin 库. 不能使用 diffuser 这种库. 可以使用 triton/flash attn 等库, 但是需要尝试导入, 如果 ImportError 就设置一个 `AVAILABLE=False`. 当需要判断是否可用的时候(需要调用的时候), 如果AVAILABLE=False那就直接raise一个error
    - modeling_xxx.py 里面用于把模型的各个组件组合起来, 组合成一个可以forward和predict action的 PreTrainedModel
- 只要安装了基础库, 那么一个模型可以完全脱离当前的项目运行. 可以直接把这个模型的文件夹拿走运行. 因此, 在模型文件夹内部, 不要import相对路径的文件

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


