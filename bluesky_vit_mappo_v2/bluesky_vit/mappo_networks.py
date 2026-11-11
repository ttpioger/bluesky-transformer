from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Normal
from .structured_vit import StructuredViT, observation_to_tokens


class SharedViTActor(nn.Module):
    """Parameter-shared Gaussian actor with tanh-squashed actions."""
    def __init__(self, n_tokens: int, action_dim: int = 2, features_dim: int = 128,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        # dropout=0.0: PPO needs the same log-prob at collection and update time.
        self.encoder = StructuredViT(n_tokens, features_dim, d_model, n_heads,
                                     n_layers, dropout=dropout)
        self.mu = nn.Linear(features_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, obs):
        feat = self.encoder(obs)
        mean = self.mu(feat)
        std = self.log_std.clamp(-5, 1).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def squashed_log_prob(dist, raw_action):
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)

    def act(self, obs, deterministic=False):
        dist = self.distribution(obs)
        raw = dist.mean if deterministic else dist.rsample()
        action, log_prob = self.squashed_log_prob(dist, raw)
        return action, raw, log_prob, dist.entropy().sum(-1)

    def evaluate_raw_actions(self, obs, raw_actions):
        dist = self.distribution(obs)
        _, log_prob = self.squashed_log_prob(dist, raw_actions)
        return log_prob, dist.entropy().sum(-1)


class CentralizedViTCritic(nn.Module):
    """Attention critic: one token per agent, self-attention across agents.

    Input: joint state [B, n_agents * obs_dim] (same interface as before).
    Output: one value per agent [B, n_agents].
    Dead agents (all-zero block) are masked out of the attention.
    """
    def __init__(self, joint_state_dim: int, n_agents: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        if joint_state_dim % n_agents:
            raise ValueError("joint_state_dim must be divisible by n_agents")
        self.n_agents = n_agents
        self.obs_dim = joint_state_dim // n_agents
        self.embed = nn.Sequential(nn.Linear(self.obs_dim, d_model), nn.GELU())
        self.agent_id = nn.Embedding(n_agents, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.value = nn.Linear(d_model, 1)

    def forward(self, joint_state):
        b = joint_state.shape[0]
        x = joint_state.view(b, self.n_agents, self.obs_dim)
        pad = x.abs().sum(-1) == 0            # True = dead agent, ignored
        pad[pad.all(dim=1), 0] = False        # avoid an all-masked row (NaN)
        ids = torch.arange(self.n_agents, device=x.device)
        z = self.embed(x) + self.agent_id(ids)[None]
        z = self.transformer(z, src_key_padding_mask=pad)
        return self.value(self.norm(z)).squeeze(-1)
