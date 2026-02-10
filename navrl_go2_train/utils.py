import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.distributions import Independent
from typing import List, Tuple
from tensordict.tensordict import TensorDict

def make_mlp(num_units: List[int]) -> nn.Sequential:
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class IndependentBeta(Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, validate_args=None):
        beta_dist = Beta(alpha, beta, validate_args=validate_args)
        super().__init__(beta_dist, 1, validate_args=validate_args)

    @property
    def mode(self):
        alpha = self.base_dist.concentration1
        beta = self.base_dist.concentration0
        return (alpha - 1) / (alpha + beta - 2)

    def log_prob(self, value):
        value_clipped = value.clamp(1e-6, 1.0 - 1e-6)
        return super().log_prob(value_clipped)


class BetaActor(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)

        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


class ValueNorm(nn.Module):
    def __init__(self, input_shape: int, beta: float = 0.995, epsilon: float = 1e-5):
        super().__init__()
        self.input_shape = (
            torch.Size((input_shape,)) if isinstance(input_shape, int)
            else torch.Size(input_shape)
        )
        self.beta = beta
        self.epsilon = epsilon

        self.register_buffer('running_mean', torch.zeros(input_shape))
        self.register_buffer('running_mean_sq', torch.zeros(input_shape))
        self.register_buffer('debiasing_term', torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if torch.isnan(x).any():
            return

        dim = tuple(range(x.dim() - len(self.input_shape)))
        batch_mean = x.mean(dim=dim)
        batch_sq_mean = (x ** 2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        out = (x - mean) / torch.sqrt(var)
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        out = x * torch.sqrt(var) + mean
        return out


class GAE(nn.Module):
    def __init__(self, gamma: float = 0.99, lmbda: float = 0.95):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))

    def forward(
            self,
            reward: torch.Tensor,
            terminated: torch.Tensor,
            value: torch.Tensor,
            next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0

        for step in reversed(range(num_steps)):
            delta = (
                    reward[:, step]
                    + self.gamma * next_value[:, step] * not_done[:, step]
                    - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae)

        returns = advantages + value
        return advantages, returns


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)

    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)

    for indices in perm:
        yield tensordict[indices]


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_euler(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.pi / 2,
        torch.asin(sinp)
    )

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def vec_to_world(vec_body: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction_2d = direction.squeeze(1)
    yaw = torch.atan2(direction_2d[:, 1], direction_2d[:, 0])

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    x_w = vec_body[:, 0] * cos_yaw - vec_body[:, 1] * sin_yaw
    y_w = vec_body[:, 0] * sin_yaw + vec_body[:, 1] * cos_yaw
    z_w = vec_body[:, 2]

    return torch.stack([x_w, y_w, z_w], dim=-1)


def vec_to_new_frame(vec: torch.Tensor, new_frame_dir: torch.Tensor) -> torch.Tensor:
    new_frame_dir_2d = new_frame_dir.squeeze(1)
    yaw = torch.atan2(new_frame_dir_2d[:, 1], new_frame_dir_2d[:, 0])

    cos_yaw = torch.cos(yaw).unsqueeze(1)
    sin_yaw = torch.sin(yaw).unsqueeze(1)

    x_new = vec[:, :, 0] * cos_yaw + vec[:, :, 1] * sin_yaw
    y_new = -vec[:, :, 0] * sin_yaw + vec[:, :, 1] * cos_yaw
    z_new = vec[:, :, 2]

    return torch.stack([x_new, y_new, z_new], dim=-1)


def evaluate(
        env,
        policy,
        num_episodes: int = 10,
        max_steps: int = 1000,
) -> dict:
    total_rewards = []
    reach_goals = []
    collisions = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            with torch.no_grad():
                obs = policy(obs)

            obs = env.step(obs)

            episode_reward += obs["agents", "reward"].mean().item()

            if obs["done"].any():
                reach_goals.append(obs["stats", "reach_goal"].float().mean().item())
                collisions.append(obs["stats", "collision"].float().mean().item())
                break

        total_rewards.append(episode_reward)

    return {
        "eval/mean_reward": sum(total_rewards) / len(total_rewards),
        "eval/reach_goal_rate": sum(reach_goals) / max(len(reach_goals), 1),
        "eval/collision_rate": sum(collisions) / max(len(collisions), 1),
    }


class EpisodeStats:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.data = {k: [] for k in keys}

    def add(self, tensordict: TensorDict):
        if "done" in tensordict.keys(True, True):
            done = tensordict["done"]
        elif ("next", "done") in tensordict.keys(True, True):
            done = tensordict["next", "done"]
        else:
            print("[EpisodeStats.add] WARNING: No 'done' key found!")
            return

        num_done = done.sum().item()
        if num_done == 0:
            return

        done_flat = done.reshape(-1, done.shape[-1]).squeeze(-1)
        done_mask = done_flat.bool()

        collected_any = False
        for key in self.keys:
            if key in tensordict.keys(True, True):
                values = tensordict[key]

                if values.dim() > 2:
                    values_flat = values.reshape(-1, *values.shape[2:])
                else:
                    values_flat = values.reshape(-1)

                done_values = values_flat[done_mask]
                if done_values.numel() > 0:
                    self.data[key].append(done_values)
                    collected_any = True

    def pop(self) -> TensorDict:
        result = {}
        for key in self.keys:
            if self.data[key]:
                result[key] = torch.cat(self.data[key], dim=0)
            else:
                result[key] = torch.tensor([])

        self.data = {k: [] for k in self.keys}
        return TensorDict(result, batch_size=[])

    def __len__(self):
        for key in self.keys:
            if self.data[key]:
                return sum(v.shape[0] for v in self.data[key])
        return 0
