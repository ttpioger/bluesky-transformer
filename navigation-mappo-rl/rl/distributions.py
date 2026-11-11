import torch
import torch.nn as nn
from torch.distributions import Normal


class DiagGaussianDistribution:
    """
    Diagonal Gaussian distribution for continuous action spaces.

    This distribution is characterized by a diagonal covariance matrix, meaning
    that the actions in each dimension are independent.

    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> "DiagGaussianDistribution":
        """
        Creates the distribution from policy network output.

        :param mean_actions: Mean of the distribution.
        :param log_std: Log standard deviation of the distribution.
        :return: The distribution object.
        """
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        self.distribution = Normal(mean_actions, log_std.exp())
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of actions according to the distribution.

        :param actions: Actions.
        :return: Log probability of actions.
        """
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call proba_distribution first."
            )
        log_prob = self.distribution.log_prob(actions)
        # Sum along the action dimension
        return log_prob.sum(dim=-1)

    def sample(self) -> torch.Tensor:
        """
        Sample an action from the distribution.
        """
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call proba_distribution first."
            )
        return self.distribution.sample()

    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of the distribution.
        """
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call proba_distribution first."
            )
        entropy = self.distribution.entropy()
        # Sum along the action dimension
        return entropy.sum(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """
        Returns the mean of the distribution.
        """
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call proba_distribution first."
            )
        return self.distribution.mean
