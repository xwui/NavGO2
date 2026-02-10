import os
import sys
import argparse
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 NavRL Evaluation")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode")
parser.add_argument("--render", action="store_true", default=False, help="Enable rendering")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from navrl_go2_train.go2_nav_env import Go2NavRLEnv
from navrl_go2_train.go2_nav_env_cfg import Go2NavEnvCfg
from navrl_go2_train.ppo import PPO


def main():
    print("\n" + "=" * 60)
    print("Go2 NavRL Policy Evaluation")
    print("=" * 60 + "\n")

    cfg = Go2NavEnvCfg()
    cfg.num_envs = args_cli.num_envs
    cfg.headless = not args_cli.render

    print("[Eval] Creating environment...")
    env = Go2NavRLEnv(cfg, headless=not args_cli.render)
    env.eval()

    print(f"[Eval] Loading checkpoint from {args_cli.checkpoint}...")
    policy = PPO(cfg.algo, env.observation_spec, env.action_spec, cfg.device)

    checkpoint = torch.load(args_cli.checkpoint, map_location=cfg.device)
    policy.load_state_dict(checkpoint)
    print("[Eval] ✓ Policy loaded!")

    episode_rewards = []
    episode_lengths = []
    reach_goal_count = 0
    collision_count = 0

    print(f"\n[Eval] Running {args_cli.num_episodes} episodes...")

    for ep in range(args_cli.num_episodes):
        obs = env.reset()
        episode_reward = 0

        for step in range(args_cli.max_steps):

            with torch.no_grad():
                obs = policy(obs)

            obs = env.step(obs)
            episode_reward += obs["agents", "reward"].mean().item()

            if obs["done"].any():
                if obs["stats", "reach_goal"].any():
                    reach_goal_count += 1
                if obs["stats", "collision"].any():
                    collision_count += 1
                break

            if args_cli.render:
                time.sleep(0.01)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        print(f"       Episode {ep + 1}: reward={episode_reward:.2f}, length={step + 1}")

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"  Episodes:           {args_cli.num_episodes}")
    print(f"  Mean Reward:        {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"  Mean Length:        {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"  Success Rate:       {reach_goal_count / args_cli.num_episodes * 100:.1f}%")
    print(f"  Collision Rate:     {collision_count / args_cli.num_episodes * 100:.1f}%")
    print("=" * 60 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
