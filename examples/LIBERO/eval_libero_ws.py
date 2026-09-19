import argparse
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any, TextIO

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

LEROBOT_SITE_PACKAGES = os.environ.get("LEROBOT_SITE_PACKAGES", "")
LIBERO_HOME = os.environ.get("LIBERO_HOME", "")
if not LIBERO_HOME:
    raise RuntimeError("LIBERO_HOME must be set for LIBERO evaluation.")
for extra_path in (LEROBOT_SITE_PACKAGES, LIBERO_HOME):
    if extra_path and extra_path not in sys.path:
        sys.path.append(extra_path)

import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for local_path in (REPO_ROOT, THIS_DIR):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))
from libero_client import LiberoWebSocketClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate resVLA WebSocket server on LIBERO.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--execution-horizon", type=int, default=0, help="Actions executed before replanning; 0 uses the model horizon.")
    parser.add_argument("--run-name", default="libero_ws_eval")
    parser.add_argument("--task-suite-names", default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out-path", default="")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--result-file", default="", help="Optional JSON file for evaluation results.")
    parser.add_argument("--progress-file", default="", help="Optional JSONL file updated after every episode.")
    parser.add_argument("--benchmark-config-file", default="", help="Optional JSON file containing the resolved benchmark configuration.")
    return parser.parse_args()


def _binarize_gripper_open(open_val) -> np.ndarray:
    value = float(np.asarray(open_val, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def eval_suite(
    args,
    policy: LiberoWebSocketClient,
    task_suite_name: str,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, task_suite.n_tasks)
    max_steps = _max_steps(task_suite_name)

    if args.save_video:
        import imageio
        suite_output_dir = pathlib.Path(args.video_out_path) / args.run_name / task_suite_name
        suite_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        imageio = None
        suite_output_dir = None

    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(num_tasks), desc=task_suite_name):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_episodes, task_successes = 0, 0
 
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"{task_suite_name}:{task_id}", leave=False):
            policy.reset(task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            task_episodes += 1
            total_episodes += 1
            replay_images = []
            done = False
            t = 0
            step = 0

            while t < max_steps + 10:
                if t < 10:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if args.save_video:
                    replay_images.append(image)

                state = np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))
                raw_action = policy.step(image, wrist_image, state, task_description, step)
                delta_action = np.concatenate([
                    np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1),
                    np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1),
                    _binarize_gripper_open(raw_action["open_gripper"]),
                ])
                obs, reward, done, info = env.step(delta_action.tolist())
                t += 1
                step += 1
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            if progress_stream is not None:
                print(
                    json.dumps({
                        "task_suite": task_suite_name,
                        "task_id": task_id,
                        "task_description": task_description,
                        "episode": episode_idx + 1,
                        "policy_steps": step,
                        "env_steps": t,
                        "episode_success": bool(done),
                        "episodes": total_episodes,
                        "successes": total_successes,
                        "success_rate": total_successes / total_episodes,
                        "task_episodes": task_episodes,
                        "task_successes": task_successes,
                        "task_success_rate": task_successes / task_episodes,
                    }),
                    file=progress_stream,
                    flush=True,
                )

            if args.save_video:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    suite_output_dir / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

        task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_successes / max(task_episodes, 1),
        })

    return {
        "task_suite": task_suite_name,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / max(total_episodes, 1),
        "tasks": task_results,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    np.random.seed(args.seed)
    policy = LiberoWebSocketClient(
        host=args.host,
        port=args.port,
        execution_horizon=args.execution_horizon or None,
    )
    if args.benchmark_config_file:
        config_path = pathlib.Path(args.benchmark_config_file).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump({
                "arguments": vars(args),
                "resolved": {
                    "prediction_horizon": policy.prediction_horizon,
                    "execution_horizon": policy.execution_horizon,
                },
                "server_metadata": policy.client.metadata,
            }, f, indent=2)
    suites = [suite.strip() for suite in args.task_suite_names.split(",") if suite.strip()]
    progress_stream = None
    try:
        if args.progress_file:
            progress_path = pathlib.Path(args.progress_file).expanduser()
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_stream = progress_path.open("w", encoding="utf-8", buffering=1)
        results = [
            eval_suite(args, policy, suite, progress_stream=progress_stream)
            for suite in suites
        ]
    finally:
        if progress_stream is not None:
            progress_stream.close()
    if args.result_file:
        result_path = pathlib.Path(args.result_file).expanduser()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_name": args.run_name,
                "results": results,
            }, f, indent=2)
    for result in results:
        logging.info("FINAL %s: %s", result["task_suite"], json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
