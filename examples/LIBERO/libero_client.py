import logging
import os
import time
from typing import Optional

import numpy as np
import websockets.sync.client

from resVLA.deployment import protocol


class WebsocketVLAClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 10093, timeout: float = 3000):
        self.uri = f"ws://{host}:{port}"
        self.packer = protocol.Packer()
        self.ws, self.metadata = self._wait_for_server(timeout)

    def _wait_for_server(self, timeout: float):
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Failed to connect to model server within {timeout} seconds: {self.uri}")
            try:
                conn = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=30,
                    ping_interval=60,
                    ping_timeout=60,
                )
                metadata = protocol.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Waiting for model server at %s", self.uri)
                time.sleep(2)

    def close(self):
        self.ws.close()

    def predict_action(self, payload: dict, request_id: str = "default") -> dict:
        message = {"type": "predict_action", "request_id": request_id, "payload": payload}
        self.ws.send(self.packer.pack(message))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        decoded = protocol.unpackb(response)
        if not decoded.get("ok", False):
            raise RuntimeError(f"Inference failed: {decoded.get('error')}")
        return decoded["data"]


class LiberoWebSocketClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        embodiment_tag: str = "franka",
        execution_horizon: int | None = None,
    ):
        self.client = WebsocketVLAClient(host=host, port=port)
        self.embodiment_tag = embodiment_tag
        self.prediction_horizon = int(self.client.metadata["action_horizon"])
        self.execution_horizon = (
            self.prediction_horizon
            if execution_horizon is None
            else int(execution_horizon)
        )
        if not 1 <= self.execution_horizon <= self.prediction_horizon:
            raise ValueError(
                "execution_horizon must be between 1 and the model prediction horizon "
                f"({self.prediction_horizon}), got {self.execution_horizon}."
            )
        self.task_description: Optional[str] = None
        self.actions = None
        self.action_index = 0

    def reset(self, task_description: str):
        self.task_description = task_description
        self.actions = None
        self.action_index = 0

    def close(self):
        self.client.close()

    def step(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, task_description: str, step: int) -> dict:
        if task_description != self.task_description:
            self.reset(task_description)

        if self.actions is None or self.action_index >= self.execution_horizon:
            payload = {
                "images": {
                    "cam_high": np.ascontiguousarray(image, dtype=np.uint8),
                    "cam_wrist": np.ascontiguousarray(wrist_image, dtype=np.uint8),
                },
                "state": np.asarray(state, dtype=np.float32),
                "prompt": task_description,
                "embodiment_tag": self.embodiment_tag,
                "execution_horizon": self.execution_horizon,
            }
            data = self.client.predict_action(payload, request_id=f"{step}")
            self.actions = np.asarray(data["actions"], dtype=np.float32)
            if self.actions.ndim != 2 or self.actions.shape[0] < self.execution_horizon:
                raise ValueError(
                    "Policy returned an invalid action chunk: expected at least "
                    f"{self.execution_horizon} steps, got shape {self.actions.shape}."
                )
            self.action_index = 0

        action = self.actions[self.action_index, :7]
        self.action_index += 1
        return {
            "world_vector": np.asarray(action[:3], dtype=np.float32),
            "rotation_delta": np.asarray(action[3:6], dtype=np.float32),
            "open_gripper": np.asarray(action[6:7], dtype=np.float32),
        }
