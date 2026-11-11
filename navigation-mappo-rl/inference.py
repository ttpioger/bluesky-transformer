from collections import defaultdict
from PIL import Image
import yaml
from nav.environment import Environment
import supersuit as ss
from moviepy import ImageSequenceClip
import os
import time
import random
import numpy as np
import json


def make_eval_env(
    config,
    history_length,
    render_mode="rgb_array",
    avoid_collision_checks=False,
    get_original_env=False,
):
    og_env = Environment(
        config, render_mode=render_mode, avoid_collision_checks=avoid_collision_checks
    )
    eval_env = ss.frame_stack_v1(og_env, history_length)
    eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    if get_original_env:
        return eval_env, og_env
    return eval_env


def inference(
    env,
    model,
    video_path=None,
    num_episodes=5,
    mode="rgb_array",
    og_env=None,
    window=None,
):
    """
    Run inference with a trained model.

    Args:
        model: A PPO model
        config_path: Path to the config YAML file
        video_path: Optional path to save video. If None, no video is saved.
        window: Optional SimulationWindow object for human rendering.

    Returns:
        tuple: (episode_reward, episode_length)
    """

    frames = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        while True:
            if window and window.paused:
                window.dispatch_events()
                time.sleep(1 / 60)  # prevent high CPU usage
                continue

            action = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_done = (terminated | truncated).all()
            if episode_done:
                break

            if video_path and mode == "rgb_array":
                frame = env.render()
                frames.append(frame)

            if mode == "human":
                time.sleep(1 / 30)

    env.close()
    if video_path is not None:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path)
    del env

if __name__ == "__main__":
    from rl.mappo import MAPPO
    import sys

    model_id = sys.argv[1]

    if len(sys.argv) > 2:
        env_path = sys.argv[2]
    else:
        env_path = f"{model_id}/env.yaml"
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Environment config file not found at {env_path}")

    mode = "human"  # "rgb_array"  # "human"
    model_path = f"{model_id}/best_model"  # "./models/mm1/best_model/best_model.zip"

    print(f"Loading model from {model_path}")
    print(f"Loading environment config from {env_path}")

    config = yaml.safe_load(open(env_path))
    video_path = f"movies/state_maps_fourcross.mp4"  # Path for state map videos
    env, og_env = make_eval_env(
        config,
        4,
        render_mode="human",
        avoid_collision_checks=True,
        get_original_env=True,
    )
    window = og_env.get_window()  # Get the window object
    model = MAPPO.load_model(model_path)

    inference(
        env,
        model,
        video_path,
        num_episodes=1,
        window=window,
        og_env=og_env,
    )
