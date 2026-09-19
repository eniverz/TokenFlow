import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupConstantCosineLR(LambdaLR):
    """Linear warmup, constant learning rate, then cosine decay.

    The same multiplier is applied to every optimizer parameter group.  Since
    ``LambdaLR`` scales each group's own base learning rate, group-specific
    learning-rate ratios are preserved throughout the schedule.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int,
        constant_end_ratio: float = 0.5,
        lr_min_ratio: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        self.num_training_steps = int(num_training_steps)
        self.num_warmup_steps = int(num_warmup_steps)
        self.constant_end_ratio = float(constant_end_ratio)
        self.lr_min_ratio = float(lr_min_ratio)
        self.num_cycles = float(num_cycles)
        self.constant_end_step = int(self.constant_end_ratio * self.num_training_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))

            if current_step < self.constant_end_step:
                return 1.0

            progress = float(current_step - self.constant_end_step) / float(
                max(1, self.num_training_steps - self.constant_end_step)
            )
            progress = min(1.0, progress)
            cosine = max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * 2.0 * self.num_cycles * progress))
            )
            return self.lr_min_ratio + cosine * (1.0 - self.lr_min_ratio)

        super().__init__(
            optimizer,
            lr_lambda=[lr_lambda] * len(optimizer.param_groups),
            last_epoch=last_epoch,
        )
