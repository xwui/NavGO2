from isaaclab.terrains.height_field.utils import height_field_to_mesh
import numpy as np
import time

@height_field_to_mesh
def uniform_discrete_obstacles_terrain(difficulty: float, cfg) -> np.ndarray:
    np.random.seed(cfg.seed) 
    def is_good_position(obs_list, obs_pos, min_dist):
        for i in range(len(obs_list)):
            obs_pos_i = obs_list[i]
            dist = ((obs_pos[0] - obs_pos_i[0])**2 + (obs_pos[1] - obs_pos_i[1])**2)**0.5
            if (dist < min_dist):
                return False
        return True

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obs_height_min = cfg.obstacle_height_range[0]
    obs_height_max = cfg.obstacle_height_range[1]
    obs_width_min = cfg.obstacle_width_range[0]
    obs_width_max = cfg.obstacle_width_range[1]
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    border_pixels = 0
    if hasattr(cfg, 'border_width') and cfg.border_width > 0:
        border_pixels = int(cfg.border_width / cfg.horizontal_scale)

    obs_x_range = np.arange(border_pixels, width_pixels - border_pixels, 4)
    obs_y_range = np.arange(border_pixels, length_pixels - border_pixels, 4)
    hf_raw = np.zeros((width_pixels, length_pixels))
    obs_dist = cfg.obstacles_distance
    stop_sampling = False
    obs_list = cfg.avoid_positions
    for _ in range(cfg.num_obstacles):
        height = int(np.random.uniform(obs_height_min, obs_height_max) / cfg.vertical_scale)
        width = int(np.random.uniform(obs_width_min, obs_width_max) / cfg.horizontal_scale)
        length = int(np.random.uniform(obs_width_min, obs_width_max) / cfg.horizontal_scale)
        start_time = time.time()
        good_position = False
        while (not good_position):
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            x_scale = x_start * cfg.horizontal_scale
            y_scale = y_start * cfg.horizontal_scale
            good_position = is_good_position(obs_list, [x_scale, y_scale], obs_dist)
            sample_time = time.time() 
            if (sample_time - start_time) > 0.2:
                stop_sampling = True
                break
        if (stop_sampling):
            break

        obs_list.append([x_scale, y_scale])
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    if border_pixels > 0:
        hf_raw[:border_pixels, :] = 0   
        hf_raw[-border_pixels:, :] = 0  
        hf_raw[:, :border_pixels] = 0   
        hf_raw[:, -border_pixels:] = 0  

    return np.rint(hf_raw).astype(np.int16)

