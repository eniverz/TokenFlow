import os
import time

from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import PreTrainedModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from resVLA.tools.overwatch import initialize_overwatch

logger = initialize_overwatch(__name__)


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        dataloader: DataLoader,
        accelerator: Accelerator,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        wandb_cfg: dict,
        *,
        output_dir: str,
        num_train_steps: int,
        log_interval: int,
        save_interval: int,
    ):
        self.num_train_steps = num_train_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.output_dir = output_dir

        self.model = model
        self.dataloader = dataloader
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # The processor is owned by BatchCollator, but it is part of the
        # checkpoint input contract and must be saved with the model.
        self.processor = getattr(getattr(dataloader, "collate_fn", None), "processor", None)

        self._init_wandb(wandb_cfg)
        self.print_parameters()

        self.model, self.dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.dataloader, self.optimizer, self.lr_scheduler
        )

        # TODO: add support for resuming training from a checkpoint
        # weight will be loaded before create Trainer
        self.step = 0
        self.data_length = len(self.dataloader)
        self.epoch = 0

    def _init_wandb(self, wandb_cfg):
        kwargs = OmegaConf.to_container(wandb_cfg, resolve=True)
        project_name = kwargs.pop("project")
        self.accelerator.init_trackers(
            project_name,
            init_kwargs={"wandb": kwargs}
        )

    def print_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params / 1e9:.4f}B")
        logger.info(f"Trainable parameters: {trainable_params / 1e9:.4f}B")
        logger.info(f"Trainable parameters ratio: {trainable_params / total_params * 100:.2f}%")

    def _train_step(self, batch):
        with self.accelerator.accumulate(self.model):
            outputs = self.model(batch)

            loss = outputs["loss"]
            self.accelerator.backward(loss)

            self.optimizer.step()
            if self.accelerator.sync_gradients and not self.accelerator.optimizer_step_was_skipped:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        metrics = {k: self.accelerator.reduce(v.detach(), reduction="mean").item() for k, v in outputs.items()}

        return metrics

    def train(self):
        pbar = tqdm(
            total=self.num_train_steps,
            initial=self.step,
            dynamic_ncols=True,
            disable=not self.accelerator.is_main_process
        )

        metrics_list = []

        data_iter = iter(self.dataloader)

        while self.step < self.num_train_steps:
            s_time = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                self.dataloader.dataset.set_epoch(self.epoch)
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            data_time = time.time() - s_time

            s_time = time.time()
            metrics = self._train_step(batch)
            model_time = time.time() - s_time

            metrics.update({"time/data_time": data_time, "time/model_time": model_time})
            metrics_list.append(metrics)

            if self.accelerator.sync_gradients:
                pbar.update(1)
                self.step += 1
                if self.accelerator.is_main_process:
                    pbar.set_postfix({
                        "data time": f"{data_time:.2f}",
                        "model time": f"{model_time:.2f}",
                        "loss": f"{metrics['loss']:.4f}",
                    })

                if self.step % self.log_interval == 0:
                    mean_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
                    mean_metrics.update(self._lr_metrics())
                    mean_metrics.update({"epoch": self.step / self.data_length})
                    self.accelerator.log(mean_metrics, step=self.step)
                    logger.info(f"Step {self.step}: " + "{" + ', '.join(f'{k}: {v:.4f}' for k, v in mean_metrics.items()) + "}")
                    metrics_list.clear()

                if self.step > 0 and self.step % self.save_interval == 0:
                    self._save_checkpoint()

        self._finalize()

    def _lr_metrics(self):
        learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        metrics = {"learning_rate/default": learning_rates[0]} if learning_rates else {}
        for idx, (group, lr) in enumerate(zip(self.optimizer.param_groups, learning_rates)):
            name = group.get("name", f"group_{idx}")
            metrics[f"learning_rate/{name}"] = lr
        return metrics

    def _save_checkpoint(self):
        checkpoint_path = os.path.join(self.output_dir, "checkpoint", f"step_{self.step}")
        optimizer_path = os.path.join(self.output_dir, "optimizer", f"step_{self.step}")
        
        os.makedirs(optimizer_path, exist_ok=True)

        state_dict = self.accelerator.get_state_dict(self.model)

        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_path, exist_ok=True)

            self.accelerator.unwrap_model(self.model).save_pretrained(
                checkpoint_path,
                state_dict=state_dict,
                save_function=self.accelerator.save,
                safe_serialization=True
            )

            if self.processor is not None:
                self.processor.save_pretrained(checkpoint_path)
                logger.info(f"Processor saved at {checkpoint_path}")
            else:
                logger.warning(
                    "Checkpoint has no processor: the DataLoader collate_fn does not "
                    "expose a processor."
                )

            torch.save(
                {
                    "step": self.step,
                    "epoch": self.epoch,
                    "scheduler": self.lr_scheduler.state_dict(),
                },
                os.path.join(optimizer_path, "scheduler.pt")
            )

        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(optimizer_path, f"rank_{self.accelerator.process_index}.pt")
        )

        self.accelerator.wait_for_everyone()

        logger.info(f"Checkpoint saved at step {self.step} to {checkpoint_path}")

    def _finalize(self):
        if self.step % self.save_interval != 0:
            self._save_checkpoint()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        logger.info(f"Training completed at step {self.step}. Finalized accelerator.")
