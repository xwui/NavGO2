# 接口文档

>**项目**: `isaac-go2-ros2` — 基于NavRL的 Unitree Go2 强化学习训练
>**环境**: IsaacSim 4.5 + IsaacLab 2.1

## 1.总览

```python
NavGO2/
│
├── navrl_go2_train/             # 训练模块
│   ├── __init__.py              
│   ├── go2_nav_env.py           # 环境类 Go2NavRLEnv
│   ├── dynamic_obstacles.py     # 动态障碍物管理器 + 观测计算
│   ├── ppo.py                   # PPO 策略网络
│   ├── train.py                 # 训练入口 (Hydra + WandB + SyncDataCollector)
│   ├── eval.py                  # 评估入口
│   ├── utils.py                 # 工具: MLP构建、Beta分布、GAE、ValueNorm 等
│   └── cfg/                     # Hydra 配置文件
│       ├── train.yaml           # 主训练配置
│       ├── go2.yaml             # 机器人模型 & 传感器参数
│       ├── ppo.yaml             # PPO 算法超参数
│       ├── sim.yaml             # 仿真物理引擎参数
│       ├── train_stage1.yaml    # 课程阶段1: 无障碍
│       ├── train_stage2.yaml    # 课程阶段2: 稀疏静态障碍
│       ├── train_stage3.yaml    # 课程阶段3: 混合障碍
│       └── train_stage4.yaml    # 课程阶段4: 密集全障碍
│
├── go2/                         # 机器人控制模块
│   ├── __init__.py/             
│   ├── go2_env.py               # Isaac Lab 场景定义 & RL 环境配置类
│   ├── go2_ctrl.py              # 速度指令接口 & RSL 底层策略加载
│   ├── go2_ctrl_cfg.py          # RSL-RL 策略超参数
│   └── go2_sensors.py           # 传感器管理器
│
├── env/                         # 地形
│   ├── sim_env.py               # 预设环境创建函数
│   ├── terrain.py               # 地形高度场生成算法
│   └── terrain_cfg.py           # 地形配置数据类
│
├── cfg/
│   └── sim.yaml                 # 全局仿真参数 (用于独立运行)
│
└── ckpts/                       # 预训练模型权重
    └── unitree_go2/
        ├── flat_model_6800.pt   
        ├── rough_model_7850.pt  
        └── exported/
            ├── policy.onnx      
            └── policy.pt        
```

## 2.环境接口：`Go2NavRLEnv`

>go2_nav_env.py

### 2.1 reset()

```python
def reset(self, env_ids: torch.Tensor = None) -> TensorDictBase
```

| 参数      | 类型           | 默认值 | 说明                                |
| :-------- | :------------- | :----- | :---------------------------------- |
| `env_ids` | `torch.Tensor` | `None` | 需重置的环境索引。`None` 时重置全部 |

**返回**一个`TensorDict`

>1. 重置底层物理环境
>2. **在地图边缘随机生成目标点** 
>3. **机器人朝向目标**: 重置后机器人面朝目标方向
>4. 清零所有状态缓冲

### 2.2 step()

```python
def step(self, tensordict: TensorDictBase) -> TensorDictBase
```

**输入 (Action)**

| 键                     | 形状            | 类型      | 范围      | 说明           |
| :--------------------- | :-------------- | :-------- | :-------- | :------------- |
| `("agents", "action")` | `(num_envs, 3)` | `float32` | `[-1, 1]` | 归一化速度指令 |

**动作各维度语义**:

| Index | 含义            | 映射                       | 范围                |
| :---: | :-------------- | :------------------------- | :------------------ |
|   0   | 前向线速度 `vx` | `action[0] × vx_max`       | `[-1.0, 1.0]` m/s   |
|   1   | 横向线速度 `vy` | `action[1] × vy_max`       | `[-0.2, 0.2]` m/s   |
|   2   | 偏航角速度 `ωz` | `action[2] × yaw_rate_max` | `[-0.5, 0.5]` rad/s |

**返回TensorDict：**

| 键                                              | 形状                   | 类型      | 说明               |
| :---------------------------------------------- | :--------------------- | :-------- | :----------------- |
| `("agents", "observation", "state")`            | `(num_envs, 6)`        | `float32` | 导航状态向量       |
| `("agents", "observation", "lidar")`            | `(num_envs, 1, 36, 4)` | `float32` | 激光雷达深度图     |
| `("agents", "observation", "direction")`        | `(num_envs, 1, 2)`     | `float32` | 目标方向向量       |
| `("agents", "observation", "dynamic_obstacle")` | `(num_envs, 1, 5, 6)`  | `float32` | 动态障碍物特征     |
| `("agents", "reward")`                          | `(num_envs, 1)`        | `float32` | 即时奖励           |
| `("done")`                                      | `(num_envs, 1)`        | `bool`    | 回合是否结束       |
| `("terminated")`                                | `(num_envs, 1)`        | `bool`    | 是否因终止条件结束 |
| `("truncated")`                                 | `(num_envs, 1)`        | `bool`    | 是否因超时截断     |
| `("stats")`                                     | /                      | `float32` | 回合统计           |
| `("next")`                                      | /                      | —         | 下一帧观察         |

>#### `lidar` — 激光雷达 `(num_envs, 1, 36, 4)`
>
>| 维度       |    大小    | 说明          |
>| :--------- | :--------: | :------------ |
>| Batch      | `num_envs` | 并行环境数    |
>| Channel    |    `1`     | 单通道 (距离) |
>| Horizontal |    `36`    | 水平线束数    |
>| Vertical   |    `4`     | 垂直线束      |
>
>#### `dynamic_obstacle` — 动态障碍物 `(num_envs, 1, 5, 6)`
>
>| Index | 含义     |   坐标系   | 说明                                                |
>| :---: | :------- | :--------: | :-------------------------------------------------- |
>|   0   | 方向 x   | Goal Frame | 归一化相对方向                                      |
>|   1   | 方向 y   | Goal Frame | 归一化相对方向                                      |
>|   2   | 距离     |     —      | clamp at `lidar_range`                              |
>|   3   | 速度 vx  | Goal Frame | 障碍物运动速度                                      |
>|   4   | 速度 vy  | Goal Frame | 障碍物运动速度                                      |
>|   5   | 宽度类别 |     —      | `[0, 1]`，按 `width / dyn_obs_width_res / 4` 归一化 |

### 2.3 奖励函数

`_compute_reward()` ：

| 奖励项   | 公式                           |  权重   | 说明                   |
| :------- | :----------------------------- | :-----: | :--------------------- |
| 速度奖励 | `dot(robot_vel_2d, goal_dir)`  |  `1.0`  | 向目标方向移动越快越好 |
| 朝向奖励 | `dot(robot_forward, goal_dir)` |  `1.5`  | 面朝目标方向           |
| 静态避障 | `mean(log(lidar_dist))`        |  `0.5`  | 离静态障碍越远越好     |
| 动态避障 | `mean(log(dyn_obs_dist))`      |  `0.5`  | 离动态障碍越远越好     |
| 存活奖励 | 常数 `1.0`                     |  `1.0`  | 每步固定奖励           |
| 平滑惩罚 | `‖vel - prev_vel‖`             | `-0.1`  | 惩罚速度突变           |
| 到达目标 | 在终止检查时添加               | `+15.0` | 到达目标的额外奖励     |

### 2.4 终止条件

 `_check_termination()`：

| 条件            | 判定标准                                        | 类型         |
| :-------------- | :---------------------------------------------- | :----------- |
| **到达目标**    | `‖robot_pos - target_pos‖ < 0.5m`               | `terminated` |
| **碰撞 (静态)** | LiDAR 最小读数 `< 0.35m`（忽略 `<0.2m` 的噪声） | `terminated` |
| **碰撞 (动态)** | 与动态障碍物距离 `< 0.5m`                       | `terminated` |
| **出界**        | `|x| > air_wall` 或 `|y| > air_wall`            | `terminated` |
| **跌倒**        | 机器人高度 `< 0.25m`                            | `terminated` |
| **超时**        | `progress_buf >= max_episode_length`            | `truncated`  |

>初始化20步稳定

## 3.PPO算法

### 1.网络结构

```python
输入:
  ├── lidar (num_envs, 1, 36, 4)
  │     └─→ Conv2d(4→16, 3层) → Flatten → Linear(128) → LayerNorm
  │           输出: _cnn_feature (128,)
  │
  ├── dynamic_obstacle (num_envs, 1, 5, 6)
  │     └─→ Flatten → MLP[128, 64]
  │           输出: _dyn_feature (64,)
  │
  └── state (num_envs, 6)

拼接: [_cnn_feature, state, _dyn_feature] → (128+6+64 = 198,)
  └─→ MLP[256, 256]
        输出: _feature (256,)

Actor:  _feature → BetaActor → Alpha,Beta → Beta分布 → action (3,)
Critic: _feature → Linear → state_value (1,)
```

