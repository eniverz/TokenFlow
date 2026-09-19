import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

from resVLA.deployment import protocol


class WebsocketModelServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 10093, idle_timeout: int = -1, metadata=None):
        self.policy = policy
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.metadata = metadata or {}
        self.last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            if self.idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        while True:
            await asyncio.sleep(5)
            if time.time() - self.last_active > self.idle_timeout:
                logging.info("Idle timeout reached; shutting down websocket server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket):
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = protocol.Packer()
        await websocket.send(packer.pack(self.metadata))

        while True:
            try:
                message = protocol.unpackb(await websocket.recv())
                self.last_active = time.time()
                response = self._route_message(message)
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _route_message(self, message):
        if not isinstance(message, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": "default",
                "error": {"message": f"Message must be a dict, got {type(message)}"},
            }

        request_id = message.get("request_id", "default")
        message_type = message.get("type", "predict_action")
        payload = message.get("payload", message)

        if message_type == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": request_id}

        if message_type not in {"predict_action", "infer"}:
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": request_id,
                "error": {"message": f"Unsupported message type: {message_type}"},
            }

        try:
            data = self.policy.predict_action(**payload)
        except Exception as exc:
            logging.exception("Policy inference failed for request_id=%s", request_id)
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": request_id,
                "error": {"message": str(exc)},
            }

        return {
            "status": "ok",
            "ok": True,
            "type": "inference_result",
            "request_id": request_id,
            "data": data,
        }
