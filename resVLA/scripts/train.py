from transformers import AutoModel
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from resVLA.tools import register_all
from resVLA.tools.overwatch import initialize_overwatch

logger = initialize_overwatch(__name__)


def train(cfg: DictConfig) -> None:
    logger.info("Starting preparing...")
    model_cfg = instantiate(cfg.model)
    model = AutoModel.from_config(model_cfg)
    logger.info("Model created.")

    weight_loader = instantiate(cfg.weight_loader)
    model = weight_loader.load(model)
    logger.info("Model weights loaded.")

    optimizer = instantiate(cfg.optimizer, model=model)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    logger.info("Optimizer and scheduler created.")

    trainer = instantiate(
        cfg.train,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    logger.info("Trainer created. Starting training...")
    trainer.train()


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    resume = cfg.get("resume", False)
    if resume:
        """TODO: implement resume training"""
    else:
        train(cfg)


if __name__ == "__main__":
    register_all()
    main()
