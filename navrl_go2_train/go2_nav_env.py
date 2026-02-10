import os
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2.go2_ctrl import set_vel_command, init_base_vel_cmd, get_rsl_rough_policy
from go2.go2_env import Go2RSLEnvCfg

from dynamic_obstacles import DynamicObstacleManager, compute_dynamic_obstacle_observation

from omegaconf import DictConfig


class Go2NavRLEnv:

    def __init__(self, cfg: DictConfig, headless: bool = True):
        self.cfg = cfg
        self.device = cfg.device
        self.common_step_counter = 0
        self.num_envs = cfg.num_envs
        self.max_episode_length = cfg.max_episode_length
        self.headless = headless
        self.dt = cfg.sim_dt * cfg.decimation
        self._init_go2_env()
        self._init_navigation_components()
        self._set_specs()
        self.stats = self.stats_spec.zero()
        self.common_step_counter = 0
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.training = True
        print(f"[Go2NavRLEnv] Initialized with {self.num_envs} environments")

    def _init_go2_env(self):

        import math

        from go2.go2_env import Go2NavSimCfg, Go2RSLEnvCfg
        go2_env_cfg = Go2RSLEnvCfg()

        go2_env_cfg.scene = Go2NavSimCfg(num_envs=self.num_envs, env_spacing=2.0)

        if hasattr(self.cfg.env, 'num_obstacles'):
            num_obstacles = self.cfg.env.num_obstacles
            print(f"[Go2NavRL] Setting static obstacle count: {num_obstacles}")

            if hasattr(go2_env_cfg.scene, 'terrain') and go2_env_cfg.scene.terrain is not None:
                terrain_gen = go2_env_cfg.scene.terrain.terrain_generator
                if hasattr(terrain_gen, 'sub_terrains') and 'obstacles' in terrain_gen.sub_terrains:
                    terrain_gen.sub_terrains['obstacles'].num_obstacles = num_obstacles
                    print(f"[Go2NavRL] Terrain obstacle count{num_obstacles}")

        go2_env_cfg.decimation = math.ceil(1.0 / go2_env_cfg.sim.dt / 50.0)
        go2_env_cfg.sim.render_interval = go2_env_cfg.decimation

        init_base_vel_cmd(self.num_envs, device=self.device)

        print("[Go2NavRLEnv] Creating environment with Go2NavSimCfg (obstacle terrain integrated)...")
        self.go2_env, self.low_level_policy = get_rsl_rough_policy(go2_env_cfg)

        self.scene = self.go2_env.unwrapped.scene
        self.robot = self.scene["unitree_go2"]
        print("[Go2NavRLEnv] ✓ Environment created with obstacle terrain at /World/ground")

        self.go2_obs, _ = self.go2_env.reset()
        print(f"[Go2NavRLEnv] ✓ go2_obs initialized with shape {self.go2_obs.shape}")

    def _init_navigation_components(self):

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 3)
            self.target_dir = torch.zeros(self.num_envs, 3)
            self.goal_yaw = torch.zeros(self.num_envs)
            self.last_action = torch.zeros(self.num_envs, 3)
            self.prev_vel = torch.zeros(self.num_envs, 3)
            self.reward = torch.zeros(self.num_envs, 1)
            self.terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.truncated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.prev_distance = torch.zeros(self.num_envs, 1)

        self.lidar_range = self.cfg.sensor.lidar_range
        self.lidar_hbeams = self.cfg.sensor.lidar_hbeams
        self.lidar_vbeams = self.cfg.sensor.lidar_vbeams

        print(
            f"[NavGO2] LiDAR Config: range={self.lidar_range}, hbeams={self.lidar_hbeams}, vbeams={self.lidar_vbeams}")

        map_range = float(self.cfg.env.get('map_range', 15.0))

        self.obstacle_range = map_range
        self.spawn_edge = map_range + 4.0
        self.air_wall = map_range + 10.0
        self.terrain_size = map_range * 2 + 20
        self.terrain_border = 5.0
        self.map_range = [self.air_wall, self.air_wall, 4.5]

        if hasattr(self, 'go2_env') and self.go2_env is not None:
            unwrapped_env = self.go2_env.unwrapped
            unwrapped_env.spawn_edge = self.spawn_edge

        print(
            f"[NavGO2] Boundaries: map_range={map_range}m (obs=±{self.obstacle_range}, spawn=±{self.spawn_edge}, wall=±{self.air_wall})")

        if self.cfg.env_dyn.num_obstacles > 0:
            self._init_dynamic_obstacles()

    def _init_dynamic_obstacles(self):
        """Create dynamic obstacle manager and visual prims."""

        num_dyn_obs = self.cfg.env_dyn.num_obstacles

        vel_range = list(self.cfg.env_dyn.get('vel_range', [0.5, 1.5]))
        local_range = list(self.cfg.env_dyn.get('local_range', [3.0, 3.0]))
        enable_vis = self.cfg.env_dyn.get('enable_visualization', True)
        max_obs_width = self.cfg.env_dyn.get('max_obs_width', 1.0)
        max_obs_height = self.cfg.env_dyn.get('max_obs_height', 2.0)

        spawn_range = [self.obstacle_range, self.obstacle_range]

        print(f"[NavGO2] Dynamic Obstacles: N={num_dyn_obs}, range=±{spawn_range[0]}m, vis={enable_vis}")

        self.dyn_obs_manager = DynamicObstacleManager(
            num_obstacles=num_dyn_obs,
            spawn_range=spawn_range,
            vel_range=tuple(vel_range),
            local_range=local_range,
            sim_dt=self.dt,
            device=self.device,
            enable_visualization=enable_vis,
            max_obs_width=max_obs_width,
            max_obs_height=max_obs_height,
        )

        self.dyn_obs_manager.create_obstacles()

        # Set up state tensor references
        self.dyn_obs_state = self.dyn_obs_manager.dyn_obs_state
        self.dyn_obs_pos = self.dyn_obs_manager.dyn_obs_pos
        self.dyn_obs_goal = self.dyn_obs_manager.dyn_obs_goal
        self.dyn_obs_origin = self.dyn_obs_manager.dyn_obs_origin
        self.dyn_obs_vel = self.dyn_obs_manager.dyn_obs_vel
        self.dyn_obs_size = self.dyn_obs_manager.dyn_obs_size
        self.dyn_obs_step_count = 0
        self.dyn_obs_width_res = self.dyn_obs_manager.dyn_obs_width_res

        print(f"[Go2NavRL] DynamicObstacleManager initialized successfully")

    def _set_specs(self):

        observation_dim = 6

        num_dim_each_dyn_obs_state = 6
        dyn_obs_num = self.cfg.algo.feature_extractor.dyn_obs_num

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),

                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams),
                                                           device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 2), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, dyn_obs_num, num_dim_each_dyn_obs_state),
                                                                      device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": BoundedTensorSpec(-1.0, 1.0, (3,), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)

        self.stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        self.info_spec = CompositeSpec({
            "robot_state": UnboundedContinuousTensorSpec((13,), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec

    @property
    def batch_size(self):
        return torch.Size([self.num_envs])

    def train(self):

        self.training = True
        return self

    def eval(self):

        self.training = False
        return self

    def reset(self, env_ids: torch.Tensor = None) -> TensorDictBase:

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.go2_obs, _ = self.go2_env.reset()

        if isinstance(self.go2_obs, dict):
            if "policy" in self.go2_obs:
                self.go2_obs = self.go2_obs["policy"]
        elif isinstance(self.go2_obs, tuple):
            self.go2_obs = self.go2_obs[0]

        from go2.go2_ctrl import set_vel_command
        zero_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        set_vel_command(zero_cmd)

        self.go2_obs[:, 9:12] = zero_cmd

        self._reset_target(env_ids)

        pos = self.robot.data.root_pos_w[:, :3]
        diff = self.target_pos - pos
        facing_yaw = torch.atan2(diff[:, 1], diff[:, 0])
        facing_quat = self._yaw_to_quat(facing_yaw)

        current_root_state = self.robot.data.default_root_state.clone()
        current_root_state[:, :3] = pos
        current_root_state[:, 3:7] = facing_quat
        self.robot.write_root_state_to_sim(current_root_state)
        self.robot.update(dt=0.0)

        self.stats[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.prev_vel[env_ids] = 0.0
        self.smooth_action = torch.zeros(self.num_envs, 3, device=self.device)
        self.terminated[env_ids] = False
        self.truncated[env_ids] = False

        robot_pos_2d = self.robot.data.root_pos_w[env_ids, :2]
        target_pos_2d = self.target_pos[env_ids, :2]
        self.prev_distance[env_ids] = (target_pos_2d - robot_pos_2d).norm(dim=-1, keepdim=True)

        return self._compute_observation()

    def _reset_target(self, env_ids: torch.Tensor):

        import numpy as np
        num_reset = env_ids.shape[0]

        EDGE_POS = self.spawn_edge
        RANDOM_RANGE = self.spawn_edge

        if self.training:

            masks = torch.tensor([
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
            ], dtype=torch.float, device=self.device)

            shifts = torch.tensor([
                [0., EDGE_POS, 0.],
                [0., -EDGE_POS, 0.],
                [EDGE_POS, 0., 0.],
                [-EDGE_POS, 0., 0.],
            ], dtype=torch.float, device=self.device)

            mask_indices = np.random.randint(0, masks.size(0), size=num_reset)
            selected_masks = masks[mask_indices]
            selected_shifts = shifts[mask_indices]

            target_pos = (2 * RANDOM_RANGE) * torch.rand(num_reset, 3, device=self.device) - RANDOM_RANGE
            target_pos[:, 2] = 0.0

            target_pos = target_pos * selected_masks + selected_shifts

            self.target_pos[env_ids] = target_pos
        else:

            masks = torch.tensor([
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
            ], dtype=torch.float, device=self.device)

            shifts = torch.tensor([
                [0., EDGE_POS, 0.],
                [0., -EDGE_POS, 0.],
                [EDGE_POS, 0., 0.],
                [-EDGE_POS, 0., 0.],
            ], dtype=torch.float, device=self.device)

            mask_indices = np.random.randint(0, masks.size(0), size=num_reset)
            selected_masks = masks[mask_indices]
            selected_shifts = shifts[mask_indices]

            target_pos = (2 * RANDOM_RANGE) * torch.rand(num_reset, 3, device=self.device) - RANDOM_RANGE
            target_pos[:, 2] = 0.0
            target_pos = target_pos * selected_masks + selected_shifts

            self.target_pos[env_ids] = target_pos

        robot_pos = self.robot.data.root_pos_w[env_ids, :3]
        self.target_dir[env_ids] = self.target_pos[env_ids] - robot_pos

        rpos_2d = self.target_dir[env_ids, :2]
        self.goal_yaw[env_ids] = torch.atan2(rpos_2d[:, 1], rpos_2d[:, 0])

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:

        high_level_action = tensordict[("agents", "action")].clone().detach()

        self.last_action = high_level_action.clone()

        vx_body = high_level_action[:, 0] * self.cfg.limits.vx_max
        vy_body = high_level_action[:, 1] * self.cfg.limits.vy_max
        yaw_rate = high_level_action[:, 2] * self.cfg.limits.yaw_rate_max

        raw_scaled_action = torch.stack([vx_body, vy_body, yaw_rate], dim=-1)

        action_to_execute = raw_scaled_action

        robot_quat = self.robot.data.root_quat_w
        robot_yaw = self._quat_to_yaw(robot_quat)
        robot_pos_2d = self.robot.data.root_pos_w[:, :2]
        target_pos_2d = self.target_pos[:, :2]
        rpos_2d = target_pos_2d - robot_pos_2d
        goal_yaw = torch.atan2(rpos_2d[:, 1], rpos_2d[:, 0])
        heading_error = self._normalize_angle(goal_yaw - robot_yaw)
        self.goal_yaw = goal_yaw

        set_vel_command(action_to_execute)

        self.go2_obs[:, 9:12] = action_to_execute.clone()

        with torch.inference_mode():
            low_level_action = self.low_level_policy(self.go2_obs)

        if low_level_action.abs().max() > 10.0:
            print(f"[Go2NavRL] Warning: Exploding RSL Action! Max={low_level_action.abs().max():.2f}. Clamping.")
            low_level_action = low_level_action.clamp(-5.0, 5.0)

        self.go2_obs, _, _, _ = self.go2_env.step(low_level_action)

        if self.cfg.env_dyn.num_obstacles > 0:
            self._move_dynamic_obstacles()

        obs_td = self._compute_observation()
        self._compute_reward()

        self.progress_buf += 1

        self._check_termination()

        self.stats["return"] += self.reward

        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        self.stats["truncated"] = self.truncated.float()

        done_snapshot = (self.terminated | self.truncated).clone()
        terminated_snapshot = self.terminated.clone()
        truncated_snapshot = self.truncated.clone()
        stats_snapshot = self.stats.clone()

        done_ids = torch.where(done_snapshot.squeeze(-1))[0]
        if len(done_ids) > 0:
            self._reset_idx(done_ids)

        return TensorDict({
            "agents": TensorDict({
                "observation": obs_td[("agents", "observation")],
                "reward": self.reward.clone(),
            }, [self.num_envs]),
            "done": done_snapshot,
            "terminated": terminated_snapshot,
            "truncated": truncated_snapshot,
            "stats": stats_snapshot,
            "next": obs_td,
        }, self.batch_size)

    def _reset_idx(self, env_ids: torch.Tensor):

        if len(env_ids) == 0:
            return

        self.go2_env.unwrapped._reset_idx(env_ids)

        self.robot.update(dt=0.0)

        if hasattr(self.scene, 'sensors'):
            height_scanner = self.scene.sensors.get("height_scanner")
            if height_scanner is not None:
                height_scanner.update(dt=0.0)

            nav_lidar = self.scene.sensors.get("nav_lidar")
            if nav_lidar is not None:
                nav_lidar.update(dt=0.0)

        action_manager = self.go2_env.unwrapped.action_manager
        if hasattr(action_manager, 'reset'):
            action_manager.reset(env_ids)
        if hasattr(action_manager, '_action'):
            action_manager._action[env_ids] = 0.0

        with torch.inference_mode():
            new_obs_dict = self.go2_env.unwrapped.observation_manager.compute()
            if "policy" in new_obs_dict:
                self.go2_obs[env_ids] = new_obs_dict["policy"][env_ids]

        self.go2_obs[env_ids, 36:48] = 0.0

        from go2.go2_ctrl import set_vel_command
        zero_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        set_vel_command(zero_cmd)

        self.go2_obs[env_ids, 9:12] = 0.0

        self.robot.update(dt=0.0)
        pos = self.robot.data.root_pos_w[env_ids]

        self.terminated[env_ids] = False
        self.truncated[env_ids] = False
        self.stats[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.prev_vel[env_ids] = 0.0
        self.smooth_action[env_ids] = 0.0

        self._reset_target(env_ids)

        robot_pos_2d = pos[:, :2]
        target_pos_2d = self.target_pos[env_ids, :2]
        self.prev_distance[env_ids] = (target_pos_2d - robot_pos_2d).norm(dim=-1, keepdim=True)

        self.target_dir[env_ids] = self.target_pos[env_ids] - pos[:, :3]

        diff = self.target_pos[env_ids] - pos[:, :3]
        facing_yaw = torch.atan2(diff[:, 1], diff[:, 0])

        facing_quat = self._yaw_to_quat(facing_yaw)

        current_root_state = self.robot.data.default_root_state[env_ids].clone()
        current_root_state[:, :3] = pos[:, :3]
        current_root_state[:, 3:7] = facing_quat
        self.robot.write_root_state_to_sim(current_root_state, env_ids)
        self.robot.update(dt=0.0)

        from go2.go2_ctrl import set_vel_command
        zero_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        set_vel_command(zero_cmd)

        with torch.inference_mode():

            new_obs_dict = self.go2_env.unwrapped.observation_manager.compute()

            if "policy" in new_obs_dict:
                self.go2_obs[env_ids] = new_obs_dict["policy"][env_ids]
            else:
                print("[Go2NavRL] Warning: 'policy' not found in observation_manager.compute()")

        self.go2_obs[env_ids, 9:12] = 0.0

    def _update_target_direction(self, env_ids: torch.Tensor):

        current_pos = self.robot.data.root_pos_w[env_ids, :3]

        self.target_dir[env_ids] = self.target_pos[env_ids] - current_pos

        self.prev_vel[env_ids] = 0.0

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:

        half_yaw = yaw * 0.5
        w = torch.cos(half_yaw)
        x = torch.zeros_like(yaw)
        y = torch.zeros_like(yaw)
        z = torch.sin(half_yaw)
        return torch.stack([w, x, y, z], dim=-1)

    def _compute_observation(self) -> TensorDictBase:

        robot_pos_2d = self.robot.data.root_pos_w[:, :2]
        robot_quat = self.robot.data.root_quat_w
        robot_vel_2d = self.robot.data.root_lin_vel_w[:, :2]

        current_yaw = self._quat_to_yaw(robot_quat)

        target_pos_2d = self.target_pos[:, :2]
        rpos_2d = target_pos_2d - robot_pos_2d

        distance = rpos_2d.norm(dim=-1, keepdim=True)

        goal_yaw = torch.atan2(rpos_2d[:, 1], rpos_2d[:, 0])

        rpos_body_2d = self._world_to_body_2d(rpos_2d, current_yaw)
        vel_body_2d = self._world_to_body_2d(robot_vel_2d, current_yaw)

        rpos_body_dir = rpos_body_2d / distance.clamp(min=1e-6)

        heading_error = self._normalize_angle(goal_yaw - current_yaw)

        state = torch.cat([
            rpos_body_dir,
            distance,
            heading_error.unsqueeze(-1),
            vel_body_2d,
        ], dim=-1)

        lidar_scan = self._get_lidar_scan()

        lidar_proximity = self.lidar_range - lidar_scan.clamp(max=self.lidar_range)

        lidar_scan = lidar_proximity.reshape(self.num_envs, 1, self.lidar_hbeams, self.lidar_vbeams)

        target_dir_2d = rpos_2d / distance.clamp(min=1e-6)

        self.goal_yaw = goal_yaw

        robot_pos_3d = self.robot.data.root_pos_w[:, :3]
        dyn_obs_states = self._get_dynamic_obstacle_states(robot_pos_3d, current_yaw)

        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": state,

                    "lidar": lidar_scan,
                    "direction": target_dir_2d.unsqueeze(1),
                    "dynamic_obstacle": dyn_obs_states.unsqueeze(1),
                }, [self.num_envs]),
            }, [self.num_envs]),
            "stats": self.stats.clone(),
            "info": TensorDict({
                "env_ids": torch.arange(self.num_envs, device=self.device),
                "target_pos": self.target_pos,
                "target_yaw": goal_yaw,
                "current_pos": robot_pos_2d,
            }, [self.num_envs])
        }, [self.num_envs])

        return obs

    def _get_lidar_scan(self) -> torch.Tensor:

        lidar = self.scene.sensors["nav_lidar"]

        ray_hits = lidar.data.ray_hits_w
        pos_w = lidar.data.pos_w

        diff = ray_hits - pos_w.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)

        scan = distances.clamp(max=self.lidar_range)

        scan = scan.reshape(self.num_envs, self.lidar_hbeams, self.lidar_vbeams)

        return scan

    def _get_dynamic_obstacle_states(self, robot_pos: torch.Tensor, current_yaw: torch.Tensor) -> torch.Tensor:

        dyn_obs_num = self.cfg.algo.feature_extractor.dyn_obs_num

        if self.cfg.env_dyn.num_obstacles == 0 or not hasattr(self, 'dyn_obs_manager'):
            self.dyn_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            self.dyn_obs_distance_reward = torch.full((self.num_envs, dyn_obs_num), self.lidar_range,
                                                      device=self.device)
            return torch.zeros(self.num_envs, dyn_obs_num, 6, device=self.device)

        target_pos_2d = self.target_pos[:, :2]
        robot_pos_2d = robot_pos[:, :2]
        target_dir_2d = target_pos_2d - robot_pos_2d
        target_dir_2d = target_dir_2d / target_dir_2d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        dyn_obs_states, dyn_collision, dyn_obs_distance_reward = compute_dynamic_obstacle_observation(
            robot_pos=robot_pos,
            target_dir_2d=target_dir_2d,
            dyn_obs_pos=self.dyn_obs_pos,
            dyn_obs_vel=self.dyn_obs_vel,
            dyn_obs_size=self.dyn_obs_size,
            num_closest=dyn_obs_num,
            lidar_range=self.lidar_range,
            dyn_obs_width_res=getattr(self, 'dyn_obs_width_res', 0.25),
            device=self.device
        )

        self.dyn_collision = dyn_collision
        self.dyn_obs_distance_reward = dyn_obs_distance_reward

        return dyn_obs_states

    def _compute_reward(self):

        robot_pos = self.robot.data.root_pos_w[:, :3]
        robot_vel = self.robot.data.root_lin_vel_w[:, :3]
        robot_yaw = self._quat_to_yaw(self.robot.data.root_quat_w)

        rpos = self.target_pos - robot_pos
        distance = rpos[:, :2].norm(dim=-1, keepdim=True)
        goal_dir = rpos[:, :2] / distance.clamp(min=1e-6)

        reward_vel = (robot_vel[:, :2] * goal_dir).sum(dim=-1)

        robot_fwd = torch.stack([torch.cos(robot_yaw), torch.sin(robot_yaw)], dim=-1)
        reward_facing = (robot_fwd * goal_dir).sum(dim=-1)

        lidar_scan = self._get_lidar_scan()
        lidar_filtered = torch.where(
            lidar_scan > 0.2,
            lidar_scan,
            torch.full_like(lidar_scan, self.lidar_range)
        )
        reward_safety_static = torch.log(lidar_filtered.clamp(min=1e-6)).mean(dim=(1, 2))

        if hasattr(self, 'dyn_obs_distance_reward') and self.cfg.env_dyn.num_obstacles > 0:
            reward_safety_dynamic = torch.log(
                self.dyn_obs_distance_reward.clamp(min=1e-6, max=self.lidar_range)
            ).mean(dim=-1)
        else:
            reward_safety_dynamic = torch.zeros(self.num_envs, device=self.device)

        penalty_smooth = (robot_vel - self.prev_vel).norm(dim=-1)

        self.reward = (
                reward_vel * 1.0 +
                reward_facing * 1.5 +
                reward_safety_static * 0.5 +
                reward_safety_dynamic * 0.5 +
                1.0 -
                penalty_smooth * 0.1
        ).unsqueeze(-1)

        self.prev_vel = robot_vel.clone()
        self.prev_distance = distance.clone()

    def _check_termination(self):

        robot_pos = self.robot.data.root_pos_w[:, :3]
        rpos = self.target_pos - robot_pos
        distance_2d = rpos[:, :2].norm(dim=-1)

        reach_goal = distance_2d < 0.5

        lidar_scan = self._get_lidar_scan()
        lidar_flat = lidar_scan.reshape(self.num_envs, -1)

        lidar_filtered = torch.where(lidar_flat > 0.2, lidar_flat,
                                     torch.full_like(lidar_flat, self.lidar_range))
        min_dist = lidar_filtered.min(dim=-1)[0]
        collision = min_dist < 0.35

        if hasattr(self, 'dyn_collision') and self.cfg.env_dyn.num_obstacles > 0:
            collision = collision | self.dyn_collision.squeeze(-1)

        collision = collision & (self.progress_buf > 20)

        out_of_bounds = (
                (robot_pos[:, 0].abs() > self.map_range[0]) |
                (robot_pos[:, 1].abs() > self.map_range[1])
        )

        out_of_bounds = out_of_bounds & (self.progress_buf > 20)

        robot_height = self.robot.data.root_pos_w[:, 2]
        is_fallen = robot_height < 0.25

        self.terminated = (reach_goal | collision | out_of_bounds | is_fallen).unsqueeze(-1)
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.reward = self.reward + reach_goal.float().unsqueeze(-1) * 15.0

        self.stats["reach_goal"] = reach_goal.float().unsqueeze(-1)
        self.stats["collision"] = collision.float().unsqueeze(-1)

        self.stats["truncated"] = self.truncated.float()

    def _move_dynamic_obstacles(self):

        if self.cfg.env_dyn.num_obstacles == 0 or not hasattr(self, 'dyn_obs_manager'):
            return

        self.dyn_obs_manager.move_obstacles()

        if self.dyn_obs_manager.dyn_obs_state is not None:
            self.dyn_obs_state = self.dyn_obs_manager.dyn_obs_state
            self.dyn_obs_vel = self.dyn_obs_manager.dyn_obs_vel
            self.dyn_obs_size = self.dyn_obs_manager.dyn_obs_size
            self.dyn_obs_origin = self.dyn_obs_manager.dyn_obs_origin
            self.dyn_obs_goal = self.dyn_obs_manager.dyn_obs_goal

    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:

        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _world_to_body(self, vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        x_b = vec_w[:, 0] * c + vec_w[:, 1] * s
        y_b = -vec_w[:, 0] * s + vec_w[:, 1] * c
        z_b = vec_w[:, 2]

        return torch.stack([x_b, y_b, z_b], dim=-1)

    def _world_to_body_2d(self, vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        x_b = vec_w[:, 0] * c + vec_w[:, 1] * s
        y_b = -vec_w[:, 0] * s + vec_w[:, 1] * c

        return torch.stack([x_b, y_b], dim=-1)

    def _world_to_goal_frame_2d(self, vec_w: torch.Tensor, goal_yaw: torch.Tensor) -> torch.Tensor:

        c = torch.cos(goal_yaw)
        s = torch.sin(goal_yaw)

        x_g = vec_w[:, 0] * c + vec_w[:, 1] * s
        y_g = -vec_w[:, 0] * s + vec_w[:, 1] * c

        return torch.stack([x_g, y_g], dim=-1)

    def _goal_frame_to_world_2d(self, vec_g: torch.Tensor, goal_yaw: torch.Tensor) -> torch.Tensor:

        c = torch.cos(goal_yaw)
        s = torch.sin(goal_yaw)

        x_w = vec_g[:, 0] * c - vec_g[:, 1] * s
        y_w = vec_g[:, 0] * s + vec_g[:, 1] * c

        return torch.stack([x_w, y_w], dim=-1)

    def _goal_frame_to_body_2d(self, vec_g: torch.Tensor, goal_yaw: torch.Tensor,
                               robot_yaw: torch.Tensor) -> torch.Tensor:

        vec_w = self._goal_frame_to_world_2d(vec_g, goal_yaw)

        vec_b = self._world_to_body_2d(vec_w, robot_yaw)
        return vec_b

    def _world_to_body_batch(self, vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:

        c = torch.cos(yaw).unsqueeze(-1)
        s = torch.sin(yaw).unsqueeze(-1)

        x_b = vec_w[:, :, 0] * c + vec_w[:, :, 1] * s
        y_b = -vec_w[:, :, 0] * s + vec_w[:, :, 1] * c
        z_b = vec_w[:, :, 2]

        return torch.stack([x_b.squeeze(-1), y_b.squeeze(-1), z_b], dim=-1)

    def set_seed(self, seed: int):

        torch.manual_seed(seed)
        np.random.seed(seed)

    def enable_render(self, enable: bool):

        pass

    def close(self):

        self.go2_env.close()
