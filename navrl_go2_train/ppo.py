import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors

from navrl_go2_train.utils import (
    ValueNorm, make_mlp, IndependentBeta, BetaActor, GAE, make_batch
)


class PPO(TensorDictModuleBase):

    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)

        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")],
                             ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature",
                       del_keys=False),
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        action_leaf_spec = action_spec["agents", "action"]
        self.action_dim = action_leaf_spec.shape[-1]
        self.n_agents = 1

        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.actor.learning_rate
        )

        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):

        lidar = tensordict.get(("agents", "observation", "lidar"), None)
        state = tensordict.get(("agents", "observation", "state"), None)
        dyn_obs = tensordict.get(("agents", "observation", "dynamic_obstacle"), None)

        has_nan_input = False

        self.feature_extractor(tensordict)

        feature = tensordict.get("_feature", None)

        self.actor(tensordict)

        alpha = tensordict.get("alpha", None)
        beta = tensordict.get("beta", None)

        self.critic(tensordict)

        action_normalized = tensordict["agents", "action_normalized"]

        actions = (2 * action_normalized * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        tensordict["agents", "action"] = actions

        return tensordict

    def train_step(self, tensordict):

        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]

        rewards = tensordict["next", "agents", "reward"]
        dones = tensordict["next", "terminated"]

        values = tensordict["state_value"]

        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)

        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)

        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()

        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict):

        self.feature_extractor(tensordict)

        feature = tensordict.get("_feature", None)

        action_dist = self.actor.get_dist(tensordict)

        alpha = tensordict.get("alpha", None)
        beta_param = tensordict.get("beta", None)

        action_normalized = tensordict[("agents", "action_normalized")]
        action_normalized = action_normalized.clamp(1e-6, 1.0 - 1e-6)
        log_probs = action_dist.log_prob(action_normalized)

        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        advantage = tensordict["adv"]

        sample_log_prob = tensordict["sample_log_prob"]

        log_ratio = log_probs - sample_log_prob
        log_ratio = log_ratio.clamp(-20., 20.)
        ratio = torch.exp(log_ratio).unsqueeze(-1)

        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1. - self.cfg.actor.clip_ratio, 1. + self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim

        b_value = tensordict["state_value"]
        ret = tensordict["ret"]
        value = self.critic(tensordict)["state_value"]
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio)
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        loss = entropy_loss + actor_loss + critic_loss

        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        feature_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=5.)

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)

        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()

        explained_var = 1 - F.mse_loss(value, ret) / ret.var().clamp(min=1e-6)

        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

    def state_dict(self):

        return {
            'feature_extractor': self.feature_extractor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'value_norm': self.value_norm.state_dict(),
        }

    def load_state_dict(self, state_dict):

        self.feature_extractor.load_state_dict(state_dict['feature_extractor'])
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.value_norm.load_state_dict(state_dict['value_norm'])
