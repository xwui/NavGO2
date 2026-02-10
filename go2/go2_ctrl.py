import os
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

GO2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(GO2_DIR)
CKPTS_DIR = os.path.join(PROJECT_ROOT, "ckpts")

base_vel_cmd_input = None
_base_vel_cmd_device = "cuda:0"  

def init_base_vel_cmd(num_envs, device="cuda:0"):
    global base_vel_cmd_input, _base_vel_cmd_device
    _base_vel_cmd_device = device
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    print(f"[Go2Ctrl] Initialized base_vel_cmd_input with shape {base_vel_cmd_input.shape} on device {device}")

def set_vel_command(commands):
    global base_vel_cmd_input
    if base_vel_cmd_input is None:
        print("[Go2Ctrl] Warning: base_vel_cmd_input not initialized yet.")
        return
    cmd = commands.clone().detach().to(base_vel_cmd_input.device)
    if cmd.shape != base_vel_cmd_input.shape:
        print(f"[Go2Ctrl] Error: Command shape mismatch. Expected {base_vel_cmd_input.shape}, got {cmd.shape}")
        return
    if torch.isnan(cmd).any():
        print("[Go2Ctrl] Warning: NaN detected in velocity command, using zeros")
        cmd = torch.zeros_like(cmd)
    base_vel_cmd_input.copy_(cmd)

def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input
    if base_vel_cmd_input is None:
        num_envs = env.num_envs
        return torch.zeros((num_envs, 3), dtype=torch.float32, device=env.device)
    return base_vel_cmd_input.clone().to(env.device)
def get_rsl_flat_policy(cfg):
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=CKPTS_DIR, 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy
def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=CKPTS_DIR, 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy