import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

from .configuration_tokenflow import TokenFlowConfig
from .modular_tokenflow import FlowRefiner


class TokenFlow(PreTrainedModel):
    config_class = TokenFlowConfig
    base_model_prefix = "stage2"

    def __init__(self, config: TokenFlowConfig):
        super().__init__(config)
        self.stage2 = AutoModel.from_config(config.stage2_config)
        self.flow_refiner = FlowRefiner(config)
        self.post_init()
        for block in self.flow_refiner.blocks:
            nn.init.zeros_(block.ffn.down.weight)
        self.stage2.requires_grad_(False)
        self.stage2.eval()

    def train(self, mode=True):
        super().train(mode)
        self.stage2.eval()
        return self

    def forward(self, samples):
        coarse_action, contexts, context_mask = self.stage2.predict_action(
            samples, num_context_layers=self.config.num_refiner_layers,
        )
        actions = samples["actions"].float()
        mask = samples["action_mask"].unsqueeze(1)
        coarse_action = coarse_action.float().masked_fill(~mask, 0)
        x_0 = (coarse_action + self.config.fm_noise_std * torch.randn_like(actions)).masked_fill(~mask, 0)
        time = torch.rand(actions.shape[0], 1, 1, device=actions.device)
        x_t = ((1 - time) * x_0 + time * actions).masked_fill(~mask, 0)
        velocity = self.flow_refiner(x_t, time, coarse_action, contexts, context_mask)
        error = (velocity.float() - (actions - x_0)).square().masked_fill(~mask, 0)
        loss = error.sum() / (mask.sum() * actions.shape[1])
        return {"loss": loss, "loss/flow_loss": loss.detach()}

    @torch.inference_mode()
    def predict_action(self, samples):
        coarse_action, contexts, context_mask = self.stage2.predict_action(
            samples, num_context_layers=self.config.num_refiner_layers,
        )
        mask = torch.tensor([
            [d in self.config.action_dims[name] for d in range(self.config.action_dim)]
            for name in samples["embodiment"]
        ], device=coarse_action.device).unsqueeze(1)
        coarse_action = coarse_action.float().masked_fill(~mask, 0)
        actions = (coarse_action + self.config.fm_noise_std * torch.randn_like(coarse_action)).masked_fill(~mask, 0)
        dt = 1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            time = torch.full((actions.shape[0], 1, 1), step * dt, device=actions.device)
            velocity = self.flow_refiner(actions, time, coarse_action, contexts, context_mask)
            actions = (actions + dt * velocity.float()).masked_fill(~mask, 0)
        return actions
