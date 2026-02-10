import os
import sys
import argparse
import datetime
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 NavRL Training")

AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments")

args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type

from navrl_go2_train.go2_nav_env import Go2NavRLEnv
from navrl_go2_train.ppo import PPO
from navrl_go2_train.utils import EpisodeStats


class SyncDataCollector:

    def __init__(
            self,
            env,
            policy,
            frames_per_batch: int,
            total_frames: int,
            device: str = "cuda:0",
    ):
        self.env = env
        self.policy = policy
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.device = device

        self._frames = 0
        self._fps = 0.0
        self._current_obs = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._frames >= self.total_frames:
            raise StopIteration

        import time
        start_time = time.time()

        data = self._collect_rollout()

        elapsed = time.time() - start_time
        self._fps = self.frames_per_batch / elapsed if elapsed > 0 else 0
        self._frames += self.frames_per_batch

        return data

    def _collect_rollout(self):

        num_steps = self.frames_per_batch // self.env.num_envs

        obs_list = []
        next_obs_list = []
        value_list = []
        log_prob_list = []
        reward_list = []
        done_list = []
        stats_list = []

        if self._current_obs is None:
            self._current_obs = self.env.reset()

        obs = self._current_obs

        for step in range(num_steps):
            with torch.no_grad():
                obs = self.policy(obs)

            obs_list.append(obs.clone())
            value_list.append(obs["state_value"].clone())
            log_prob_list.append(obs["sample_log_prob"].clone())

            next_obs_td = self.env.step(obs)

            reward_list.append(next_obs_td["agents", "reward"].clone())
            done_list.append(next_obs_td["done"].clone())

            if "stats" in next_obs_td.keys():
                stats_list.append(next_obs_td["stats"].clone())

            obs = next_obs_td["next"]
            next_obs_list.append(obs.clone())

        from tensordict import TensorDict

        rollout = TensorDict({}, batch_size=[self.env.num_envs, num_steps])

        for key in obs_list[0].keys(True, True):
            values = [o[key] for o in obs_list]
            rollout[key] = torch.stack(values, dim=1)

        rollout["state_value"] = torch.stack(value_list, dim=1)
        rollout["sample_log_prob"] = torch.stack(log_prob_list, dim=1)

        next_rollout = TensorDict({}, batch_size=[self.env.num_envs, num_steps])
        for key in next_obs_list[0].keys(True, True):
            values = [o[key] for o in next_obs_list]
            next_rollout[key] = torch.stack(values, dim=1)

        next_rollout["agents", "reward"] = torch.stack(reward_list, dim=1)
        next_rollout["terminated"] = torch.stack(done_list, dim=1)

        rollout["next"] = next_rollout
        rollout["terminated"] = next_rollout["terminated"]
        rollout["done"] = rollout["terminated"]

        if stats_list:
            for key in stats_list[0].keys(True, True):

                values = [s[key] for s in stats_list]

                stacked = torch.stack(values, dim=1)

                if isinstance(key, tuple):
                    rollout[key] = stacked
                else:
                    rollout[("stats", key)] = stacked

        self._current_obs = obs

        return rollout


CFG_PATH = os.path.join(os.path.dirname(__file__), "cfg")


@hydra.main(config_path=CFG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("[NavGO2] Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    if args_cli.num_envs is not None:
        cfg.env.num_envs = args_cli.num_envs
    if hasattr(args_cli, 'headless'):
        cfg.headless = args_cli.headless

    run_mode = "HEADLESS" if cfg.headless else "GUI"
    print(f"[NavGO2] Starting training in {run_mode} mode...")

    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        id=cfg.wandb.run_id,
        resume="must" if cfg.wandb.run_id else None,
    )

    print("[NavGO2] Creating Go2NavRL environment...")
    env = Go2NavRLEnv(cfg, headless=cfg.headless)
    env.set_seed(cfg.seed)
    env.train()

    print("[NavGO2] Creating PPO policy...")
    policy = PPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        print(f"[NavGO2] Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=cfg.device)
        policy.load_state_dict(state_dict)
        print("[NavGO2] Checkpoint loaded successfully!")

    frames_per_batch = cfg.env.num_envs * cfg.algo.training_frame_num
    collector = SyncDataCollector(
        env=env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=int(cfg.max_frame_num),
        device=cfg.device,
    )

    episode_stats_keys = [
        k for k in env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]

    episode_stats = EpisodeStats(episode_stats_keys)

    import time
    train_start_time = time.time()

    print("[NavGO2] Starting training loop...")

    for i, data in enumerate(collector):

        info = {
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
        }

        train_stats = policy.train_step(data)

        info.update(train_stats)

        for name, param in policy.named_parameters():
            if torch.isnan(param).any():
                print(f"[Go2NavRL] CRITICAL: Model parameter {name} has NaN!")
                break

        episode_stats.add(data)

        flush_threshold = min(cfg.env.num_envs, 10)

        if len(episode_stats) >= flush_threshold:
            stats = episode_stats.pop()
            for k, v in stats.items(True, True):

                key_str = ".".join(k) if isinstance(k, tuple) else k

                if v.numel() > 0:
                    mean_val = v.float().mean().item()
                    info[f"train/{key_str}"] = mean_val

        if i % cfg.eval_interval == 0:
            print(f"\n[NavGO2] Evaluating at step {i}...")
            env.eval()

            max_eval_steps = cfg.env.max_episode_length
            obs = env.reset()

            all_dones = []
            all_stats = []

            for step_idx in range(max_eval_steps):
                with torch.no_grad():
                    obs = policy(obs)
                step_result = env.step(obs)
                all_dones.append(step_result["done"].clone())
                all_stats.append(step_result["stats"].clone())
                obs = step_result["next"]

            all_dones = torch.stack(all_dones, dim=1)

            done_squeezed = all_dones.squeeze(-1)
            first_done_idx = torch.argmax(done_squeezed.long(), dim=1)

            never_done = ~done_squeezed.any(dim=1)
            first_done_idx[never_done] = max_eval_steps - 1

            eval_info = {}
            stat_keys = all_stats[0].keys()
            for key in stat_keys:
                key_data = torch.stack([s[key] for s in all_stats], dim=1)

                idx = first_done_idx.view(-1, 1, *([1] * (key_data.dim() - 2)))
                first_done_data = torch.gather(key_data, 1, idx.expand(-1, 1, *key_data.shape[2:])).squeeze(1)
                eval_info[f"eval/stats.{key}"] = first_done_data.float().mean().item()

            env.train()
            env.reset()

            print(f"[NavGO2] Eval: "
                  f"return={eval_info.get('eval/stats.return', 0):.2f}, "
                  f"ep_len={eval_info.get('eval/stats.episode_len', 0):.1f}, "
                  f"reach_goal={eval_info.get('eval/stats.reach_goal', 0):.2%}, "
                  f"collision={eval_info.get('eval/stats.collision', 0):.2%}")

            info.update(eval_info)

        run.log(info)

        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[NavGO2] Model saved at step {i}")

        if i % 100 == 0:
            loss_info = f"actor_loss={train_stats.get('actor_loss', 0):.4f}, " \
                        f"critic_loss={train_stats.get('critic_loss', 0):.4f}, " \
                        f"entropy={train_stats.get('entropy', 0):.4f}"
            print(f"[NavGO2] Step {i}, Frames: {collector._frames}, "
                  f"FPS: {collector._fps:.1f}, {loss_info}")

    final_ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), final_ckpt_path)
    print(f"[NavGO2] Final model saved to {final_ckpt_path}")

    total_time = time.time() - train_start_time
    print(f"[NavGO2] Training completed in {total_time / 3600:.2f} hours")

    wandb.finish()
    env.close()
    simulation_app.close()
    print("[NavGO2] Training completed!")


if __name__ == "__main__":
    main()
