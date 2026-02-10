from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
import go2.go2_ctrl as go2_ctrl
import torch

from isaaclab.sensors import RayCasterCfg, patterns


def get_lidar_distance(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    device = getattr(env, 'device', 'cuda:0')
    num_envs = getattr(env, "num_envs", 1)
    default_num_rays = 180
    default_max_d = 10.0
    
    try:
        sensor = env.scene.sensors[sensor_cfg.name]
        if sensor is None:
            raise AttributeError("Sensor is None")
        if not hasattr(sensor, 'data') or sensor.data is None:
            raise AttributeError("Sensor data not available")
        data = sensor.data
        if not hasattr(data, 'ray_hits_w') or data.ray_hits_w is None:
            raise AttributeError("ray_hits_w not available")
        if not hasattr(data, 'pos_w') or data.pos_w is None:
            raise AttributeError("pos_w not available")
        diff = data.ray_hits_w - data.pos_w.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)  
        return dist.to(device)
        
    except (AttributeError, KeyError, RuntimeError, TypeError) as e:
        try:
            sensor = env.scene.sensors.get(sensor_cfg.name)
            if sensor is not None:
                if hasattr(sensor, 'num_rays'):
                    default_num_rays = sensor.num_rays
                elif hasattr(sensor, 'cfg') and hasattr(sensor.cfg, 'pattern_cfg'):
                    
                    pattern = sensor.cfg.pattern_cfg
                    if hasattr(pattern, 'horizontal_res'):
                        default_num_rays = int(360 / pattern.horizontal_res)
                if hasattr(sensor, 'cfg') and hasattr(sensor.cfg, 'max_distance'):
                    default_max_d = sensor.cfg.max_distance
        except Exception:
            pass
        return torch.full((num_envs, default_num_rays), float(default_max_d), device=device)

@configclass
class Go2SimCfg(InteractiveSceneCfg):
    """Go2 仿真场景配置 (符合 Isaac Lab 4.5 规范)"""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1e-4))
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/.*_foot", 
        history_length=3, 
        track_air_time=True
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        
        
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        
        
        
        mesh_prim_paths=["/World/ground"],
        max_distance=1.5,  
        
        drift_range=(0.0, 0.0),
    )

    nav_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.2)),
        attach_yaw_only=True,
        
        
        
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_res=10.0,  
            vertical_ray_angles=[-10.0, 0.0, 10.0, 20.0]  
        ),
        debug_vis=False,     
        max_distance=4.0,    
        
        
        mesh_prim_paths=["/World"],
        
        drift_range=(0.0, 0.0),
    )


from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from env.terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg

@configclass
class Go2NavSimCfg(Go2SimCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground_plane",  
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0.001))
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(50, 50),         
            border_width=5.0,      
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            use_cache=False,
            color_scheme='height',
            sub_terrains={
                "obstacles": HfUniformDiscreteObstaclesTerrainCfg(
                    seed=0,
                    size=(50, 50),
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=5.0,  
                    obstacle_width_range=(0.5, 1.0),
                    obstacle_height_range=(1.0, 2.0),
                    num_obstacles=200,
                    obstacles_distance=2.0,
                    platform_width=0.0,
                    avoid_positions=[[0, 0]],
                ),
            },
        ),
        visual_material=None,
        max_init_terrain_level=None,
        collision_group=-1,
    )
    
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],  
        max_distance=1.5,
        drift_range=(0.0, 0.0),
    )
    
    nav_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.2)),
        attach_yaw_only=True,
        
        
        
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_res=10.0,  
            vertical_ray_angles=[-10.0, 0.0, 10.0, 20.0]  
        ),
        debug_vis=True,
        max_distance=4.0,  
        mesh_prim_paths=["/World"],
        drift_range=(0.0, 0.0),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="unitree_go2", joint_names=[".*"])

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
                               params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="unitree_go2")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        base_vel_cmd = ObsTerm(func=go2_ctrl.base_vel_cmd)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(func=mdp.height_scan,
                              params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                              clip=(-1.0, 1.0))
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class NavRLCfg(ObsGroup):

        root_pos_w = ObsTerm(
            func=mdp.root_pos_w, 
            params={"asset_cfg": SceneEntityCfg("unitree_go2")}
        )

        root_quat_w = ObsTerm(
            func=mdp.root_quat_w, 
            params={"asset_cfg": SceneEntityCfg("unitree_go2")}
        )

        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w, 
            params={"asset_cfg": SceneEntityCfg("unitree_go2")}
        )
        
        
        lidar_range = ObsTerm(
            func=get_lidar_distance, 
            params={"sensor_cfg": SceneEntityCfg("nav_lidar")}
        )
    nav_policy: NavRLCfg = NavRLCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="unitree_go2",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )

def reset_root_state_at_edges(
    env,
    env_ids: torch.Tensor,
    asset_cfg,
) -> None:

    robot = env.scene[asset_cfg.name]
    num_reset = len(env_ids)
    device = env.device
    spawn_edge = getattr(env, 'spawn_edge', 22.0)
    EDGE_POS = spawn_edge        
    LONG_RANGE = spawn_edge      
    LAT_RANGE = 1.0              
    ROBOT_HEIGHT = 0.38          

    pos_x = torch.zeros(num_reset, device=device)
    pos_y = torch.zeros(num_reset, device=device)
    edge_idx = torch.randint(0, 4, (num_reset,), device=device)
    long_pos = (torch.rand(num_reset, device=device) * 2 * LONG_RANGE) - LONG_RANGE
    lat_pos = (torch.rand(num_reset, device=device) * 2 * LAT_RANGE) - LAT_RANGE
    mask_n = (edge_idx == 0)
    pos_x[mask_n] = long_pos[mask_n]
    pos_y[mask_n] = EDGE_POS + lat_pos[mask_n]
    mask_s = (edge_idx == 1)
    pos_x[mask_s] = long_pos[mask_s]
    pos_y[mask_s] = -EDGE_POS + lat_pos[mask_s]
    mask_e = (edge_idx == 2)
    pos_x[mask_e] = EDGE_POS + lat_pos[mask_e]
    pos_y[mask_e] = long_pos[mask_e]
    mask_w = (edge_idx == 3)
    pos_x[mask_w] = -EDGE_POS + lat_pos[mask_w]
    pos_y[mask_w] = long_pos[mask_w]
    pos = torch.stack([pos_x, pos_y, torch.full_like(pos_x, ROBOT_HEIGHT)], dim=1)

    if num_reset <= 4:  
        for i in range(num_reset):
            env_id = env_ids[i].item()
            print(f"[DEBUG] Env {env_id} spawn at X={pos[i,0]:.2f}, Y={pos[i,1]:.2f}, Z={pos[i,2]:.2f}")

    center = torch.zeros(num_reset, 3, device=device)
    diff = center - pos
    facing_yaw = torch.atan2(diff[:, 1], diff[:, 0])
    half_yaw = facing_yaw * 0.5
    quat = torch.zeros(num_reset, 4, device=device)
    quat[:, 0] = torch.cos(half_yaw)  
    quat[:, 3] = torch.sin(half_yaw)
    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] = pos
    default_root_state[:, 3:7] = quat
    default_root_state[:, 7:10] = 0.0  
    default_root_state[:, 10:13] = 0.0
    robot.write_root_state_to_sim(default_root_state, env_ids)
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)
    robot.reset(env_ids)
    zero_velocity = torch.zeros(len(env_ids), 6, device=device)  
    robot.write_root_velocity_to_sim(zero_velocity, env_ids)

    
    if hasattr(robot, 'set_joint_effort_target'):
        zero_efforts = torch.zeros(len(env_ids), robot.num_joints, device=device)
        robot.set_joint_effort_target(zero_efforts, env_ids=env_ids)
@configclass
class EventCfg:
    """Configuration for events."""
    reset_base = mdp.EventTermCfg(
        func=reset_root_state_at_edges,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("unitree_go2")},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class Go2RSLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 environment."""
    scene = Go2SimCfg(num_envs=2, env_spacing=2.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.viewer.eye = [0.0, 40.0, 40.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        self.decimation = 4
        self.sim.dt = 0.005  
        self.sim.render_interval = self.decimation  
        self.sim.gravity = (0.0, 0.0, -9.81)         
        self.sim.disable_contact_processing = False  
        self.sim.render.antialiasing_mode = None
        self.episode_length_s = 20.0 
        self.is_finite_horizon = False
        self.actions.joint_pos.scale = 0.25
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
@configclass
class Go2NavRLEnvCfg(ManagerBasedRLEnvCfg):
    scene = Go2NavSimCfg(num_envs=4, env_spacing=2.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()
    def __post_init__(self):
        self.viewer.eye = [0.0, 40.0, 40.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation  
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.disable_contact_processing = False
        self.sim.render.antialiasing_mode = None
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.actions.joint_pos.scale = 0.25
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

def camera_follow(env):
    if (env.unwrapped.scene.num_envs == 1):
        robot_position = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                                robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler('zyx')[0]
        yaw_rotation = R.from_euler('z', yaw).as_matrix()
        set_camera_view(
            yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
            robot_position
        )


