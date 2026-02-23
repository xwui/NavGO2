"""
动态障碍物 - VisualizationMarkers
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import time

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


class DynamicObstacleManager:

    def __init__(
        self,
        num_obstacles: int,
        spawn_range: List[float],
        vel_range: Tuple[float, float],
        local_range: List[float],
        sim_dt: float,
        device: str = "cuda:0",
        enable_visualization: bool = True,
        max_obs_width: float = 1.0,
        max_obs_height: float = 2.0,
        **kwargs
    ):
        self.spawn_range = spawn_range[:2]
        self.vel_range = vel_range
        self.local_range = local_range[:2] if len(local_range) >= 2 else local_range
        self.dt = sim_dt
        self.device = device
        self.enable_visualization = enable_visualization

        self.N_w = 4
        self.max_obs_width = max_obs_width
        self.max_obs_height = max_obs_height
        self.dyn_obs_width_res = self.max_obs_width / float(self.N_w)
        self.max_obs_3d_height = max_obs_height

        self.dyn_obs_num_of_each_category = max(1, int(num_obstacles / self.N_w))
        self.num_obstacles = self.dyn_obs_num_of_each_category * self.N_w

        self.dyn_obs_pos = None
        self.dyn_obs_goal = None
        self.dyn_obs_origin = None
        self.dyn_obs_vel = None
        self.dyn_obs_size = None
        self.dyn_obs_vel_norm = None
        self.dyn_obs_state = None

        self.vis_markers = None
        self.marker_indices = None
        self.dyn_obs_step_count = 0

    def create_obstacles(self):
        if self.num_obstacles == 0:
            return

        self.dyn_obs_pos = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_goal = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_origin = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_vel = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_size = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_vel_norm = torch.zeros(self.num_obstacles, 1, dtype=torch.float, device=self.device)
        self.dyn_obs_state = torch.zeros(self.num_obstacles, 13, dtype=torch.float, device=self.device)
        self.dyn_obs_state[:, 3] = 1.0

        def check_pos_validity(prev_pos_list, curr_pos, min_dist):
            for prev_pos in prev_pos_list:
                if np.linalg.norm(curr_pos - prev_pos) <= min_dist:
                    return False
            return True

        obs_dist = 2 * np.sqrt(self.spawn_range[0] * self.spawn_range[1] / self.num_obstacles)
        curr_obs_dist = obs_dist
        prev_pos_list = []

        obstacle_idx = 0
        marker_indices_list = []

        for category_idx in range(self.N_w):
            obs_width = float(category_idx + 1) * self.max_obs_width / float(self.N_w)

            for _ in range(self.dyn_obs_num_of_each_category):
                idx = obstacle_idx
                obstacle_idx += 1

                start_time = time.time()
                while True:
                    ox = np.random.uniform(-self.spawn_range[0], self.spawn_range[0])
                    oy = np.random.uniform(-self.spawn_range[1], self.spawn_range[1])
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)

                    if time.time() - start_time > 0.1:
                        curr_obs_dist *= 0.8
                        start_time = time.time()

                    if valid:
                        prev_pos_list.append(curr_pos)
                        break

                curr_obs_dist = obs_dist

                self.dyn_obs_origin[idx] = torch.tensor([ox, oy], dtype=torch.float, device=self.device)
                self.dyn_obs_pos[idx] = torch.tensor([ox, oy], dtype=torch.float, device=self.device)
                self.dyn_obs_size[idx] = torch.tensor([obs_width, self.max_obs_height], dtype=torch.float, device=self.device)

                oz = self.max_obs_height / 2.0
                self.dyn_obs_state[idx, 0] = ox
                self.dyn_obs_state[idx, 1] = oy
                self.dyn_obs_state[idx, 2] = oz

                marker_indices_list.append(category_idx)

        self.marker_indices = torch.tensor(marker_indices_list, dtype=torch.int32, device=self.device)

        if self.enable_visualization:
            self._create_visualization()

        print(f"[DynamicObstacleManager] Created {self.num_obstacles} obstacles")

    def _create_visualization(self):
        markers_dict = {}
        for w_idx in range(self.N_w):
            width = float(w_idx + 1) * self.max_obs_width / float(self.N_w)
            markers_dict[f"cylinder_w{w_idx}"] = sim_utils.CylinderCfg(
                radius=width / 2.0,
                height=self.max_obs_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    metallic=0.2
                ),
            )

        cfg = VisualizationMarkersCfg(
            prim_path="/World/DynamicObstacleMarkers",
            markers=markers_dict,
        )
        self.vis_markers = VisualizationMarkers(cfg)
        self._update_visualization()
        print(f"[DynamicObstacleManager] VisualizationMarkers created")

    def _update_visualization(self):
        if self.vis_markers is None or self.dyn_obs_state is None:
            return
        self.vis_markers.visualize(
            translations=self.dyn_obs_state[:, :3].clone(),
            marker_indices=self.marker_indices
        )

    def move_obstacles(self):
        if self.num_obstacles == 0 or self.dyn_obs_pos is None:
            return

        if self.dyn_obs_step_count != 0:
            dyn_obs_goal_dist = torch.norm(self.dyn_obs_pos - self.dyn_obs_goal, dim=1)
        else:
            dyn_obs_goal_dist = torch.zeros(self.num_obstacles, device=self.device)

        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5

        num_new_goal = torch.sum(dyn_obs_new_goal_mask).item()
        if num_new_goal > 0:
            sample_x = -self.local_range[0] + 2.0 * self.local_range[0] * torch.rand(
                int(num_new_goal), dtype=torch.float, device=self.device)
            sample_y = -self.local_range[1] + 2.0 * self.local_range[1] * torch.rand(
                int(num_new_goal), dtype=torch.float, device=self.device)
            self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + \
                torch.stack([sample_x, sample_y], dim=1)

        self.dyn_obs_goal[:, 0].clamp_(-self.spawn_range[0], self.spawn_range[0])
        self.dyn_obs_goal[:, 1].clamp_(-self.spawn_range[1], self.spawn_range[1])

        if self.dyn_obs_step_count % int(2.0 / self.dt) == 0:
            self.dyn_obs_vel_norm = self.vel_range[0] + (self.vel_range[1] - self.vel_range[0]) * \
                torch.rand(self.num_obstacles, 1, dtype=torch.float, device=self.device)
            goal_dir = self.dyn_obs_goal - self.dyn_obs_pos
            goal_dist = torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1e-6)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * (goal_dir / goal_dist)
        self.dyn_obs_pos += self.dyn_obs_vel * self.dt
        self.dyn_obs_pos[:, 0].clamp_(-self.spawn_range[0], self.spawn_range[0])
        self.dyn_obs_pos[:, 1].clamp_(-self.spawn_range[1], self.spawn_range[1])

        self.dyn_obs_state[:, 0] = self.dyn_obs_pos[:, 0]
        self.dyn_obs_state[:, 1] = self.dyn_obs_pos[:, 1]

        self._update_visualization()

        self.dyn_obs_step_count += 1

    def reset(self):
        if self.dyn_obs_pos is not None:
            self.dyn_obs_pos[:] = self.dyn_obs_origin.clone()
            self.dyn_obs_state[:, 0] = self.dyn_obs_pos[:, 0]
            self.dyn_obs_state[:, 1] = self.dyn_obs_pos[:, 1]
            self.dyn_obs_goal.zero_()
            self.dyn_obs_vel.zero_()
            self.dyn_obs_step_count = 0
            self._update_visualization()

    def set_visibility(self, visible: bool):
        if self.vis_markers is not None:
            self.vis_markers.set_visibility(visible)

    def get_obstacle_info(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.dyn_obs_pos, self.dyn_obs_vel, self.dyn_obs_size


def compute_dynamic_obstacle_observation(
    robot_pos: torch.Tensor,
    target_dir_2d: torch.Tensor,
    dyn_obs_pos: torch.Tensor,
    dyn_obs_vel: torch.Tensor,
    dyn_obs_size: torch.Tensor,
    num_closest: int = 5,
    lidar_range: float = 10.0,
    dyn_obs_width_res: float = 0.25,
    device: str = "cuda:0",
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    num_envs = robot_pos.shape[0]
    num_obstacles = dyn_obs_pos.shape[0] if dyn_obs_pos is not None else 0

    dyn_obs_states = torch.zeros(num_envs, num_closest, 6, dtype=torch.float, device=device)
    dyn_obs_states[:, :, 2] = lidar_range
    dyn_collision = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
    dyn_obs_distance_reward = torch.full((num_envs, num_closest), lidar_range, dtype=torch.float, device=device)

    if num_obstacles == 0 or dyn_obs_pos is None:
        return dyn_obs_states, dyn_collision, dyn_obs_distance_reward

    robot_pos_2d = robot_pos[:, :2]

    rel_pos = dyn_obs_pos.unsqueeze(0) - robot_pos_2d.unsqueeze(1)  # [num_envs, num_obstacles, 2]
    distances = torch.norm(rel_pos, dim=-1)                          # [num_envs, num_obstacles]


    num_closest_actual = min(num_closest, num_obstacles)
    _, closest_idx = torch.topk(distances, num_closest_actual, dim=1, largest=False)

    closest_distances = distances.gather(1, closest_idx)
    out_of_range = closest_distances > lidar_range

    closest_rpos = torch.gather(rel_pos, 1, closest_idx.unsqueeze(-1).expand(-1, -1, 2))
    closest_vel = dyn_obs_vel[closest_idx]
    closest_width = dyn_obs_size[closest_idx, 0]  # [num_envs, num_closest_actual]

    cos_g = target_dir_2d[:, 0:1].unsqueeze(1)  # [num_envs, 1, 1]
    sin_g = target_dir_2d[:, 1:2].unsqueeze(1)

    rpos_gx = closest_rpos[:, :, 0] * cos_g.squeeze(-1) + closest_rpos[:, :, 1] * sin_g.squeeze(-1)
    rpos_gy = -closest_rpos[:, :, 0] * sin_g.squeeze(-1) + closest_rpos[:, :, 1] * cos_g.squeeze(-1)

    vel_gx = closest_vel[:, :, 0] * cos_g.squeeze(-1) + closest_vel[:, :, 1] * sin_g.squeeze(-1)
    vel_gy = -closest_vel[:, :, 0] * sin_g.squeeze(-1) + closest_vel[:, :, 1] * cos_g.squeeze(-1)

    dist_clamped = closest_distances.clamp(min=1e-6)
    dir_gx = rpos_gx / dist_clamped
    dir_gy = rpos_gy / dist_clamped

    width_category = (closest_width / dyn_obs_width_res - 1.0).clamp(0, 3)

    dyn_obs_states[:, :num_closest_actual, 0] = dir_gx
    dyn_obs_states[:, :num_closest_actual, 1] = dir_gy
    dyn_obs_states[:, :num_closest_actual, 2] = closest_distances.clamp(max=lidar_range)
    dyn_obs_states[:, :num_closest_actual, 3] = vel_gx
    dyn_obs_states[:, :num_closest_actual, 4] = vel_gy
    dyn_obs_states[:, :num_closest_actual, 5] = width_category

    dyn_obs_states[:, :num_closest_actual][out_of_range] = 0.0

    robot_radius = 0.3
    collision_each = closest_distances <= (closest_width / 2.0 + robot_radius)
    collision_each[out_of_range] = False
    dyn_collision = collision_each.any(dim=1, keepdim=True)

    dyn_obs_distance_reward[:, :num_closest_actual] = closest_rpos.norm(dim=-1) - closest_width / 2.0
    dyn_obs_distance_reward[:, :num_closest_actual][out_of_range] = lidar_range

    return dyn_obs_states, dyn_collision, dyn_obs_distance_reward