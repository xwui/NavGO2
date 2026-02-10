import torch
import numpy as np
import time
from typing import List, Tuple, Optional, Any


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

        # State tensors
        self.dyn_obs_pos = None
        self.dyn_obs_goal = None
        self.dyn_obs_origin = None
        self.dyn_obs_vel = None
        self.dyn_obs_size = None
        self.dyn_obs_vel_norm = None
        self.dyn_obs_state = None  # [num_obstacles, 13] (pos3 + quat4 + vel6)
        self.marker_indices = None

        self.obstacle_widths: List[float] = []
        self.dyn_obs_step_count = 0

        # USD visual prim paths (for camera visibility)
        self._visual_prim_paths: List[str] = []
        self._visual_created = False

    def create_obstacles(self):
        """Generate obstacle positions, sizes, and velocities in tensors."""
        if self.num_obstacles == 0:
            print("[DynamicObstacleManager] num_obstacles=0, skipping creation")
            return

        # Initialize state tensors
        self.dyn_obs_pos = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_goal = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_origin = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_vel = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_size = torch.zeros(self.num_obstacles, 2, dtype=torch.float, device=self.device)
        self.dyn_obs_vel_norm = torch.zeros(self.num_obstacles, 1, dtype=torch.float, device=self.device)
        self.dyn_obs_state = torch.zeros(self.num_obstacles, 13, dtype=torch.float, device=self.device)
        self.dyn_obs_state[:, 3] = 1.0  # quaternion w=1

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
        self.obstacle_widths = []

        for category_idx in range(self.N_w):
            width_level = category_idx + 1
            obs_width = float(width_level) * self.max_obs_width / float(self.N_w)
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
                self.dyn_obs_size[idx] = torch.tensor([obs_width, self.max_obs_height], dtype=torch.float,
                                                      device=self.device)
                oz = self.max_obs_height / 2.0
                self.dyn_obs_state[idx, 0] = ox
                self.dyn_obs_state[idx, 1] = oy
                self.dyn_obs_state[idx, 2] = oz

                marker_indices_list.append(category_idx)
                self.obstacle_widths.append(obs_width)

        self.marker_indices = torch.tensor(marker_indices_list, dtype=torch.int32, device=self.device)

        # Create visual prims (pure USD geometry, no physics)
        if self.enable_visualization:
            self._create_visual_prims()
        print(
            f"[DynamicObstacleManager] Created {self.num_obstacles} obstacles "
            f"(vis={'ON' if self._visual_created else 'OFF'}). "
            f"Range: X=[{self.dyn_obs_origin[:, 0].min():.1f}, {self.dyn_obs_origin[:, 0].max():.1f}], "
            f"Y=[{self.dyn_obs_origin[:, 1].min():.1f}, {self.dyn_obs_origin[:, 1].max():.1f}]")

    def _create_visual_prims(self):
        """Create visual-only cylinder prims using USD API.

        These are pure geometry — no RigidBody, no Collision, no PhysX.
        NOT managed by InteractiveScene. Visible to cameras.

        IMPORTANT: Do NOT add UsdPhysics APIs here. Any rigid body added
        outside Isaac Lab's scene management corrupts PhysX GPU buffer
        indices, causing contact_sensor and scene.reset() crashes.
        """
        try:
            from pxr import UsdGeom, Gf, Sdf, Vt
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[DynamicObstacleManager] Warning: USD stage not available, skipping visual prims")
                return

            # Create parent Xform
            parent_path = "/World/DynObsVisual"
            UsdGeom.Xform.Define(stage, parent_path)
            
            self._visual_ops = []  # Cache XformOps for fast updates

            for idx in range(self.num_obstacles):
                ox = self.dyn_obs_state[idx, 0].item()
                oy = self.dyn_obs_state[idx, 1].item()
                oz = self.dyn_obs_state[idx, 2].item()
                radius = self.obstacle_widths[idx] / 2.0

                prim_path = f"{parent_path}/obs_{idx:03d}"

                # Create Xform wrapper for positioning
                xform = UsdGeom.Xform.Define(stage, prim_path)
                # Store the translate op directly for fast updates
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(ox, oy, oz))
                self._visual_ops.append(translate_op)

                # Create cylinder geometry (visual only, NO physics)
                cylinder_path = f"{prim_path}/geom"
                cylinder = UsdGeom.Cylinder.Define(stage, cylinder_path)
                cylinder.GetRadiusAttr().Set(float(radius))
                cylinder.GetHeightAttr().Set(float(self.max_obs_height))
                cylinder.GetAxisAttr().Set("Z")

                # Set green color
                green = 0.5 + 0.5 * np.random.random()
                cylinder.GetDisplayColorAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(0.0, float(green), 0.0)])
                )

                self._visual_prim_paths.append(prim_path)

            self._visual_created = True
            print(f"[DynamicObstacleManager] Created {len(self._visual_prim_paths)} visual-only USD prims")

        except ImportError as e:
            print(f"[DynamicObstacleManager] Warning: USD API not available ({e}), no prims")
        except Exception as e:
            print(f"[DynamicObstacleManager] Warning: Failed to create visual prims: {e}")

    def _update_visual_poses(self):
        """Update visual prim positions via cached USD Xform translate ops."""
        if not self._visual_created:
            return
        
        try:
            from pxr import Gf
            # Direct update via cached ops - no stage queries or path lookups
            for idx, op in enumerate(self._visual_ops):
                op.Set(Gf.Vec3d(
                    self.dyn_obs_pos[idx, 0].item(),
                    self.dyn_obs_pos[idx, 1].item(),
                    self.max_obs_height / 2.0
                ))
        except Exception:
            pass

    def move_obstacles(self):
        """Move obstacles towards their goals each step."""

        if self.num_obstacles == 0 or self.dyn_obs_pos is None:
            return

        # Check which obstacles reached their goals
        if self.dyn_obs_step_count != 0:
            dyn_obs_goal_dist = torch.norm(self.dyn_obs_pos - self.dyn_obs_goal, dim=1)
        else:
            dyn_obs_goal_dist = torch.zeros(self.num_obstacles, device=self.device)

        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5

        # Assign new goals for obstacles that reached theirs
        num_new_goal = torch.sum(dyn_obs_new_goal_mask).item()
        if num_new_goal > 0:
            sample_x_local = -self.local_range[0] + 2.0 * self.local_range[0] * torch.rand(
                int(num_new_goal), dtype=torch.float, device=self.device)
            sample_y_local = -self.local_range[1] + 2.0 * self.local_range[1] * torch.rand(
                int(num_new_goal), dtype=torch.float, device=self.device)
            sample_goal_local = torch.stack([sample_x_local, sample_y_local], dim=1)
            self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local

        # Clamp goals to spawn range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0],
                                              min=-self.spawn_range[0], max=self.spawn_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1],
                                              min=-self.spawn_range[1], max=self.spawn_range[1])

        # Periodically update velocity direction
        if self.dyn_obs_step_count % int(2.0 / self.dt) == 0:
            self.dyn_obs_vel_norm = self.vel_range[0] + (self.vel_range[1] - self.vel_range[0]) * \
                                    torch.rand(self.num_obstacles, 1, dtype=torch.float, device=self.device)

            goal_dir = self.dyn_obs_goal - self.dyn_obs_pos
            goal_dist = torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1e-6)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * (goal_dir / goal_dist)

        # Update positions
        self.dyn_obs_pos += self.dyn_obs_vel * self.dt

        # Clamp to spawn range
        self.dyn_obs_pos[:, 0] = torch.clamp(self.dyn_obs_pos[:, 0],
                                             min=-self.spawn_range[0], max=self.spawn_range[0])
        self.dyn_obs_pos[:, 1] = torch.clamp(self.dyn_obs_pos[:, 1],
                                             min=-self.spawn_range[1], max=self.spawn_range[1])

        # Sync to state tensor
        self.dyn_obs_state[:, 0] = self.dyn_obs_pos[:, 0]
        self.dyn_obs_state[:, 1] = self.dyn_obs_pos[:, 1]

        # Update visual prim positions
        self._update_visual_poses()

        self.dyn_obs_step_count += 1

    def reset(self):
        """Reset all obstacles to their origin positions."""
        if self.dyn_obs_pos is not None:
            self.dyn_obs_pos[:] = self.dyn_obs_origin.clone()
            self.dyn_obs_state[:, 0] = self.dyn_obs_pos[:, 0]
            self.dyn_obs_state[:, 1] = self.dyn_obs_pos[:, 1]
            self.dyn_obs_goal.zero_()
            self.dyn_obs_vel.zero_()
            self.dyn_obs_step_count = 0
            self._update_visual_poses()

    def set_visibility(self, visible: bool):
        """No-op for now."""
        pass

    def get_obstacle_info(self):
        """Return obstacle positions, velocities, and sizes for PPO observation."""
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
        collision_threshold: float = 0.5,
        **kwargs
):
    """Compute dynamic obstacle observation for PPO policy (fully vectorized)."""
    num_envs = robot_pos.shape[0]
    num_obstacles = dyn_obs_pos.shape[0] if dyn_obs_pos is not None else 0

    dyn_obs_states = torch.zeros(num_envs, num_closest, 6, device=device)
    dyn_collision = torch.zeros(num_envs, dtype=torch.bool, device=device)
    dyn_obs_distance_reward = torch.full((num_envs, max(1, num_obstacles)), lidar_range, device=device)

    if num_obstacles == 0 or dyn_obs_pos is None:
        return dyn_obs_states, dyn_collision, dyn_obs_distance_reward

    robot_pos_2d = robot_pos[:, :2]  # [num_envs, 2]

    # Relative positions and distances: [num_envs, num_obstacles, 2] and [num_envs, num_obstacles]
    rel_pos = dyn_obs_pos.unsqueeze(0) - robot_pos_2d.unsqueeze(1)
    distances = torch.norm(rel_pos, dim=-1)
    dyn_obs_distance_reward = distances.clone()

    # Collision check
    min_distances = distances.min(dim=-1)[0]
    dyn_collision = min_distances < collision_threshold

    # Find closest obstacles per env: [num_envs, num_closest_actual]
    num_closest_actual = min(num_closest, num_obstacles)
    closest_indices = torch.argsort(distances, dim=-1)[:, :num_closest_actual]

    # Gather closest relative positions: [num_envs, num_closest_actual, 2]
    idx_2d = closest_indices.unsqueeze(-1).expand(-1, -1, 2)
    closest_rel_pos = torch.gather(rel_pos, 1, idx_2d)

    # Gather closest distances: [num_envs, num_closest_actual]
    closest_dist = torch.gather(distances, 1, closest_indices)

    # Gather closest velocities: [num_envs, num_closest_actual, 2]
    # dyn_obs_vel is [num_obstacles, 2], shared across envs
    closest_vel = dyn_obs_vel[closest_indices]  # advanced indexing → [num_envs, num_closest_actual, 2]

    # Gather closest widths: [num_envs, num_closest_actual]
    closest_width = dyn_obs_size[closest_indices, 0]  # [num_envs, num_closest_actual]

    # Goal-frame rotation: [num_envs, 1] for broadcasting
    goal_yaw = torch.atan2(target_dir_2d[:, 1], target_dir_2d[:, 0])
    cos_g = torch.cos(goal_yaw).unsqueeze(1)  # [num_envs, 1]
    sin_g = torch.sin(goal_yaw).unsqueeze(1)  # [num_envs, 1]

    # Rotate relative positions to goal frame
    rel_x = closest_rel_pos[:, :, 0]  # [num_envs, num_closest_actual]
    rel_y = closest_rel_pos[:, :, 1]
    rel_gx = rel_x * cos_g + rel_y * sin_g
    rel_gy = -rel_x * sin_g + rel_y * cos_g

    # Rotate velocities to goal frame
    vel_x = closest_vel[:, :, 0]
    vel_y = closest_vel[:, :, 1]
    vel_gx = vel_x * cos_g + vel_y * sin_g
    vel_gy = -vel_x * sin_g + vel_y * cos_g

    # Normalize direction
    dist_clamped = closest_dist.clamp(min=1e-6)
    dir_gx = rel_gx / dist_clamped
    dir_gy = rel_gy / dist_clamped

    # Width category
    width_category = (closest_width / dyn_obs_width_res).clamp(min=0, max=4) / 4.0

    # Assemble output: [num_envs, num_closest_actual, 6]
    dyn_obs_states[:, :num_closest_actual, 0] = dir_gx
    dyn_obs_states[:, :num_closest_actual, 1] = dir_gy
    dyn_obs_states[:, :num_closest_actual, 2] = closest_dist.clamp(max=lidar_range)
    dyn_obs_states[:, :num_closest_actual, 3] = vel_gx
    dyn_obs_states[:, :num_closest_actual, 4] = vel_gy
    dyn_obs_states[:, :num_closest_actual, 5] = width_category

    return dyn_obs_states, dyn_collision, dyn_obs_distance_reward
