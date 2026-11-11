import copy
import pickle
import os
import traceback
import numpy as np
from typing import List
import torch.nn.functional as F
from pydantic import BaseModel
from inference import inference, make_eval_env
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from tqdm import tqdm
from rl.distributions import DiagGaussianDistribution
from rl.rollout_buffer_with_states import RolloutBuffer, RolloutBufferSamples
import yaml

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)


class CentralizedCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(CentralizedCriticNetwork, self).__init__()
        self.query = nn.Linear(state_dim, hidden_dim)
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(state_dim, hidden_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=False
        )

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, n_agents, state):
        state = state.permute(1, 0, 2)
        query = self.query(state)
        key = self.key(state)
        value = self.value(state)

        attn_output, _ = self.mha(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)
        return self.layers(attn_output.mean(dim=1))


class DecentralizedActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_init=0.0):
        super(DecentralizedActorNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init, requires_grad=True
        )
        self.action_dist = DiagGaussianDistribution(action_dim)

    def forward(self, state):
        mu = self.model(state)

        # Expand log_std to match batch size
        log_std = self.log_std.expand_as(mu)

        return self.action_dist.proba_distribution(mu, log_std)


class MAPPO:
    def __init__(
        self,
        environment=None,
        eval_config=None,
        history_length=None,
        batch_size=128,
        buffer_size=1000,
        policy_kwargs={},
        model_dir="models",
        learning_rate=5e-4,
        n_epochs=4,
        ent_coef=0.1,
        vf_coef=0.5,
        infer=False,
        video_dir=None,
        inference_interval=10,
    ):
        if infer:
            return
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.buffer_size = buffer_size
        self.epochs = n_epochs
        self.lr = learning_rate
        self.gamma = 0.9
        self.environment = environment
        self.vf_coef = vf_coef
        self.num_trains = 0
        self.best_reward = -float("inf")
        self.model_dir = model_dir
        self.eval_config = eval_config
        self.history_length = history_length
        self.video_dir = video_dir
        self.inference_interval = inference_interval
        self.n_agents = environment.n_agents
        self.n_envs = environment.num_envs // self.n_agents
        os.makedirs(model_dir, exist_ok=True)
        self.save_env_yaml()

        self.create_network(
            environment.observation_space,
            environment.action_space,
            policy_kwargs,
        )
        self.config = {
            "observation_space": self.environment.observation_space,
            "action_space": self.environment.action_space,
            "policy_kwargs": policy_kwargs,
        }

        # Create optimizers for both networks
        preprocessing_params = list(self.preprocessing_layer.parameters())
        actor_params = list(self.actor_network.parameters())
        critic_params = list(self.critic_network.parameters())
        all_params = preprocessing_params + actor_params + critic_params
        self.optimizer = optim.Adam(all_params, lr=self.lr)

    def create_network(
        self, observation_space, action_space, policy_kwargs={}, **kwargs
    ):

        action_dim = int(action_space.shape[0])
        state_features_dim = policy_kwargs.get("state_features_dim", 256)
        self.preprocessing_layer = policy_kwargs["features_extractor_class"](
            observation_space,
            **policy_kwargs["features_extractor_kwargs"],
        )
        # Instantiate the value head
        self.critic_network = CentralizedCriticNetwork(
            state_features_dim,
            policy_kwargs.get("hidden_dim", 64),
        )

        self.actor_network = DecentralizedActorNetwork(
            state_features_dim,
            action_dim,
            policy_kwargs.get("hidden_dim", 64),
        )
        self.preprocessing_layer.to(device)
        self.actor_network.to(device)
        self.critic_network.to(device)

    def predict(self, observation, deterministic=False, return_details=False):
        observation = torch.from_numpy(observation).float().to(device)
        with torch.no_grad():
            observation_encoding = self.preprocessing_layer(observation)
            dist = self.actor_network(observation_encoding)

            if return_details:
                if not hasattr(self, 'n_envs'):
                    self.n_envs = 1
                if not hasattr(self, 'n_agents'):
                    self.n_agents = observation.shape[0]
                enc_reshaped = observation_encoding.reshape(
                    self.n_envs, self.n_agents, -1
                )
                values = self.critic_network(self.n_agents, enc_reshaped)
                values_repeated = values.repeat_interleave(repeats=self.n_agents, dim=0)

                state = observation.reshape(
                    self.n_envs, self.n_agents, -1
                ).repeat_interleave(repeats=self.n_agents, dim=0)

            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            # Compute log probabilities
            log_probs = dist.log_prob(action)
            unclipped_action = action.cpu().numpy()
            clipped_action = np.clip(unclipped_action, -1, 1)

            if return_details:
                return (
                    unclipped_action,
                    clipped_action,
                    log_probs.clone().cpu().numpy(),
                    values_repeated.clone().cpu().numpy(),
                    state.clone().cpu().numpy(),
                )
        return clipped_action

    def collect_rollouts(self, n_rollout_steps):
        idx = 0
        progress_bar = tqdm(total=n_rollout_steps, desc="Exploring")

        while idx < n_rollout_steps:
            obs, infos = self.environment.reset()
            _last_episode_starts = np.ones((self.environment.num_envs,), dtype=bool)
            while True:
                idx += 1
                self.num_steps += self.environment.num_envs
                unclipped_actions, clipped_actions, log_probs, values, states = (
                    self.predict(obs, return_details=True)
                )
                next_obs, rewards, terminated, truncated, infos = self.environment.step(
                    clipped_actions
                )
                dones = (terminated | truncated).astype(np.float32)

                self.rollout_buffer.add(
                    obs,  # type: ignore[arg-type]
                    unclipped_actions,
                    rewards,
                    _last_episode_starts,
                    values,
                    log_probs,
                    states,
                )
                obs = next_obs
                _last_episode_starts = dones

                if idx % 100 == 0:
                    progress_bar.n = idx
                    progress_bar.refresh()
                    progress_bar.set_postfix(
                        experiences=f"{idx:,}",
                        steps=f"{self.num_steps:,}",
                    )

                if idx >= n_rollout_steps:
                    break

            with torch.no_grad():
                _, _, _, values, _ = self.predict(obs, return_details=True)

            self.rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=dones
            )

    def learn(self, total_timesteps):
        self.num_steps = 0
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.environment.observation_space,
            action_space=self.environment.action_space,
            n_agents=self.n_agents,
            device=device,
            gae_lambda=0.95,
            gamma=self.gamma,
            n_envs=self.environment.num_envs,
        )

        iterations = 0

        while self.num_steps < total_timesteps:
            self.collect_rollouts(self.buffer_size)
            metrics = self.train()
            self.display_training_metrics(
                self.num_steps,
                metrics,
            )

            self.rollout_buffer.reset()
            iterations += 1

            if iterations % self.inference_interval == 0:
                inference_test_reward = self.save_model_callback()
                self.video_inference()

                print(f"Displaying training metrics")

                self.display_inference_metrics(
                    self.num_steps,
                    inference_test_reward,
                )

    def compute_loss(
        self,
        observations,
        actions,
        old_log_probs,
        advantages,
        targets,
        states,
        clip_ratio=0.2,
    ):
        observation_encoding = self.preprocessing_layer(observations)
        dist = self.actor_network(observation_encoding)

        batch_size = states.shape[0]
        states_flat = states.flatten(0, 1)
        states_encoding = self.preprocessing_layer(states_flat)
        states_encoding_reshaped = states_encoding.reshape(
            batch_size, self.n_agents, -1
        )
        predicted_state_values = self.critic_network(
            self.n_agents, states_encoding_reshaped
        )

        # Create normal distribution and compute log probabilities
        new_log_probs = dist.log_prob(actions)
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        actor_loss = -torch.mean(
            torch.min(ratio * advantages, clipped_ratio * advantages)
        )
        # Critic loss (value function) - optionally add clipping for stability
        critic_loss = F.mse_loss(
            input=predicted_state_values, target=targets.unsqueeze(-1)
        )

        # Entropy for exploration
        entropy = dist.entropy()
        entropy_loss = -torch.mean(entropy)

        total_loss = (
            actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
        )

        return total_loss, {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "mean_entropy": torch.mean(entropy).item(),
            "mean_ratio": torch.mean(ratio).item(),
        }

    def train(self):

        progress_bar = tqdm(total=self.epochs, desc="Training")

        for epoch in range(self.epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                self.optimizer.zero_grad()
                loss, metrics = self.compute_loss(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.old_log_prob,
                    rollout_data.advantages,
                    rollout_data.returns,
                    rollout_data.states,
                )
                loss.backward()
                # Optional: Add gradient clipping for stability

                preprocessing_params = list(self.preprocessing_layer.parameters())
                actor_params = list(self.actor_network.parameters())
                critic_params = list(self.critic_network.parameters())
                all_params = preprocessing_params + actor_params + critic_params
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                self.optimizer.step()
                self.num_trains += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        progress_bar.close()
        return metrics

    def display_training_metrics(
        self,
        total_experiences,
        metrics,
    ):
        """Display training progress and metrics using rich formatting"""
        console = Console()

        # Training info panel
        training_info = Table(show_header=False, box=None, padding=(0, 1))
        training_info.add_row("Steps:", f"[bold cyan]{self.num_steps:,}[/bold cyan]")
        training_info.add_row(
            "Experiences:",
            f"[bold yellow]{total_experiences:,}[/bold yellow]",
        )
        training_info.add_row(
            "Training Runs:",
            f"[bold magenta]{self.num_trains:,}[/bold magenta]",
        )

        # Metrics table
        metrics_table = Table(
            title="Training Metrics",
            show_header=True,
            header_style="bold blue",
        )
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta", justify="right")

        for k, v in metrics.items():
            # Format metric names nicely
            metric_name = k.replace("_", " ").title()
            if "loss" in k.lower():
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")
            elif "entropy" in k.lower():
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")
            elif "ratio" in k.lower():
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")
            else:
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")

        # Main training panel
        training_panel = Panel(
            training_info,
            title="Training Progress",
            border_style="blue",
            width=30,
        )
        # Print everything
        console.print()
        console.print(training_panel)
        console.print()
        console.print(metrics_table)
        console.print()

    def display_inference_metrics(self, total_experiences, inference_test_reward):
        console = Console()

        # Inference test panel
        inference_color = "green" if inference_test_reward > 0 else "red"
        inference_text = Text(
            f"{inference_test_reward:.3f}", style=f"bold {inference_color}"
        )

        inference_panel = Panel(
            inference_text,
            title="Inference Test Reward",
            border_style=inference_color,
            width=30,
        )

        # Print everything
        console.print()
        console.print(inference_panel)
        console.print()

    def save_env_yaml(self):
        with open(f"{self.model_dir}/env.yaml", "w") as f:
            yaml.dump(self.eval_config, f)

    def save_model(self, dir=None):
        save_path = os.path.join(self.model_dir, dir)
        os.makedirs(save_path, exist_ok=True)
        # Save both networks
        torch.save(
            {
                "actor_network": self.actor_network.state_dict(),
                "critic_network": self.critic_network.state_dict(),
                "preprocessing_layer": self.preprocessing_layer.state_dict(),
            },
            f"{save_path}/model.pth",
        )
        config = copy.deepcopy(self.config)
        with open(f"{save_path}/config.pkl", "wb") as f:
            pickle.dump(config, f)

        print(f"Saved model to {save_path}")

    @classmethod
    def load_model(cls, model_dir: str):
        print(f"Loading model from {model_dir}")
        config = pickle.load(open(f"{model_dir}/config.pkl", "rb"))
        model = MAPPO(infer=True)
        model.create_network(**config)
        checkpoint = torch.load(f"{model_dir}/model.pth")
        model.preprocessing_layer.load_state_dict(checkpoint["preprocessing_layer"])
        model.actor_network.load_state_dict(checkpoint["actor_network"])
        model.critic_network.load_state_dict(checkpoint["critic_network"])
        return model

    def save_model_callback(self):
        self.save_model("latest_model")
        mean_reward = self.inference_test()
        if mean_reward > self.best_reward:
            print(f"New best reward: {mean_reward}")
            self.best_reward = mean_reward
            self.save_model("best_model")

        return mean_reward

    def video_inference(self):
        print("Doing video inference")
        video_path = f"{self.video_dir}/videos/inference_{self.num_steps}.mp4"
        eval_env = make_eval_env(self.eval_config, self.history_length)
        try:
            inference(eval_env, self, num_episodes=3, video_path=video_path)
        except Exception as e:
            print(f"Error during inference: {e}")
            print(traceback.format_exc())
        eval_env.close()
        del eval_env

    def inference_test(self, n_episodes=5):
        eval_env = make_eval_env(
            self.eval_config, self.history_length, render_mode=None
        )
        total_reward = 0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            total_envs = len(obs)
            while True:
                action = self.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                terminated = (terminated | truncated).astype(np.float32)
                obs = next_obs
                if terminated.all():
                    break
                if isinstance(reward, np.ndarray):
                    total_reward = total_reward + reward.sum()
                else:
                    total_reward += reward
        eval_env.close()
        return total_reward / (n_episodes * total_envs)
