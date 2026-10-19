import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from resVLA.deployment.policy import CheckpointPolicy
from resVLA.deployment.websocket import WebsocketModelServer
from resVLA.tools.register_config_hook import register_all_config_hooks


def _get(cfg: DictConfig, name: str, default=None):
    """Read a runtime override from either the top level or ``server`` node."""

    if name in cfg:
        return cfg[name]
    server_cfg = cfg.get("server", {})
    if name in server_cfg:
        return server_cfg[name]
    return default


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    checkpoint_dir = _get(cfg, "checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError(
            "checkpoint_dir is required. Pass it as a Hydra override, for example "
            "+checkpoint_dir=results/<task_name>/checkpoint/step_10000."
        )

    host = _get(cfg, "host", "0.0.0.0")
    port = int(_get(cfg, "port", 10093))
    device = _get(cfg, "device", "cuda")
    dtype = _get(cfg, "dtype", "bf16")
    idle_timeout = int(_get(cfg, "idle_timeout", -1))
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    policy = CheckpointPolicy(
        checkpoint_dir=checkpoint_dir,
        cfg=resolved_cfg,
        device=device,
        dtype=dtype,
    )
    server = WebsocketModelServer(
        policy=policy,
        host=host,
        port=port,
        idle_timeout=idle_timeout,
        metadata=policy.metadata(),
    )

    logging.info("Serving checkpoint %s on ws://%s:%s", checkpoint_dir, host, port)
    server.serve_forever()

def _run():
    sys.argv.extend([
        "hydra.output_subdir=null",
        "hydra.run.dir=.",
        "hydra/job_logging=disabled",
    ])
    main()

if __name__ == "__main__":
    register_all_config_hooks()
    _run()
