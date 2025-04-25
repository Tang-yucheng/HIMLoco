# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

import open3d as o3d

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.gymutil import LineGeometry

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class My_WireframeBoxGeometry(LineGeometry):
    def __init__(self, xdim=1, ydim=1, zdim=1, pose=None, color=None):
        if color is None:
            color = (1, 0, 0)
        num_lines = 12
        x = 0.5 * xdim
        y = 0.5 * ydim
        z = 0.5 * zdim
        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        # top face
        verts[0][0] = (x, y, z)
        verts[0][1] = (x, y, -z)
        verts[1][0] = (-x, y, z)
        verts[1][1] = (-x, y, -z)
        verts[2][0] = (x, y, z)
        verts[2][1] = (-x, y, z)
        verts[3][0] = (x, y, -z)
        verts[3][1] = (-x, y, -z)
        # bottom face
        verts[4][0] = (x, -y, z)
        verts[4][1] = (x, -y, -z)
        verts[5][0] = (-x, -y, z)
        verts[5][1] = (-x, -y, -z)
        verts[6][0] = (x, -y, z)
        verts[6][1] = (-x, -y, z)
        verts[7][0] = (x, -y, -z)
        verts[7][1] = (-x, -y, -z)
        # verticals
        verts[8][0] = (x, y, z)
        verts[8][1] = (x, -y, z)
        verts[9][0] = (x, y, -z)
        verts[9][1] = (x, -y, -z)
        verts[10][0] = (-x, y, z)
        verts[10][1] = (-x, -y, z)
        verts[11][0] = (-x, y, -z)
        verts[11][1] = (-x, -y, -z)
        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors
    def vertices(self):
        return self.verts
    def colors(self):
        return self._colors
    def set_colors(self, color):
        self._colors.fill(color)

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.history_length = self.cfg.env.history_length
        
        self.proprioceptive_obs_buf = torch.zeros(self.num_envs, self.num_one_step_obs*self.history_length, device=self.device, dtype=torch.float)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self._voxel_drawn = False
        self._traversability_drawn = False
        self._traversability_compute_count = 0

        self._first_reset = False
        self._first_one_index = 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # print(self.common_step_counter)

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()


        # # 体素地图光线投影
        #     # 发射射线
        # ray_results = self.gym.cast_rays(self.sim, None, self.origins_tensor, self.dirs_tensor)
        #     # 获取命中点
        # hit_pos = self.origins + ray_results['fractions'].unsqueeze(-1) * self.dirs_all
        # hit_mask = ray_results['hits'].bool()
        #     # 只处理目标环境
        # env = self.envs[self.target_env_id]
        # hits = hit_mask[self.target_env_id]
        # hit_points = hit_pos[self.target_env_id][hits]
        #     # 可视化体素地图
        # for pt in hit_points:
        #     x, y, z = pt.tolist()
        #     color = gymapi.Vec3(0.0, 1.0, 0.0)  # 默认绿色
        #     if z < 0.2:
        #         color = gymapi.Vec3(1.0, 0.0, 0.0)  # 红色表示低洼或不可通行
        #     center = gymapi.Vec3(x, y, z + 0.05)
        #     extent = gymapi.Vec3(0.04, 0.04, 0.04)
        #     gymutil.draw_box(center, extent, color, self.gym, self.viewer, env)

        # compute observations, rewards, resets, ..
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)




        # 保持重置环境在15step内不记录奖励
        max_count = 15
        indices = torch.nonzero(self.reset_buf)
        self.reset_count[indices] = max_count
        self.reset_count = torch.where(self.reset_count > 0, self.reset_count - 1, torch.zeros_like(self.reset_count))
        #     # 获取第一个为 1 的元素的索引
        # indices = torch.nonzero(self.reset_buf)
        #     # 如果找到了至少一个为 1 的元素
        # if indices.numel() > 0 and self._first_reset == False:
        #     self._first_one_index = indices[0].item()  # 获取第一个为 1 的元素的索引
        #     self._first_reset = True
        #     print("find first reset!")
        # if self._first_reset:
        #     print(f'env id: {self._first_one_index}, reward: {self.rew_buf[self._first_one_index]}')
        # print(f'env id: {0}, reward: {self.rew_buf[0]}')

        # 记录可通行性
        self.record_traversability()
        # 绘制高程体素
        if self.enable_voxel_map:
            # self.draw_voxel_terrain_surface()
            # self.export_voxel_pointcloud()
            self.export_traversability_voxel()
        else:
            # self.destory_voxel_terrain_surface()
            if self._traversability_drawn:
                self._traversability_drawn = False


        


        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs
    


    
    def record_traversability(self):
        voxel_size = 0.1
        box_half = voxel_size / 2.0
        # 获取全部机器人当前投影体素
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # # 单环境统计
        # occupy_points = quat_apply_yaw(self.base_quat[0].repeat(self.base_occupy_points.shape[0], 1), self.base_occupy_points) + (self.root_states[0, :3]).unsqueeze(0)
        # # 按照成功次数统计
        # if reset_ids[0]:
        #     self.accumulate_2d_hits(occupy_points_index, self.num_fail)
        #     # print(occupy_points_index)
        # else:
        #     self.accumulate_2d_hits(occupy_points_index, self.num_succeed)

        # 全环境统计
        occupy_points = quat_apply_yaw(self.base_quat.repeat(1, self.base_occupy_points.shape[0]), self.base_occupy_points.repeat(self.base_quat.shape[0], 1, 1)) + (self.root_states[:, :3]).unsqueeze(1)
        occupy_points[:, :, 0] += self.cfg.terrain.border_size
        occupy_points[:, :, 1] += self.cfg.terrain.border_size
        occupy_points_index = occupy_points // voxel_size
        # # 按照成功次数统计
        # reset_ids = self.reset_buf * ~self.time_out_buf
        # reset_mask = reset_ids.bool()
        # if torch.any(reset_mask):
        #     self.accumulate_2d_hits(occupy_points_index[reset_mask], self.num_fail)
        # self.accumulate_2d_hits(occupy_points_index[~reset_mask], self.num_succeed)

        # 滤除重置后15step以内的环境
        mask = (self.reset_count == 0)
        rew_buf_filtered = self.rew_buf[mask]
        occupy_points_index_filtered = occupy_points_index[mask]
        # print("-----------------------------------------------------------------")
        # print(rew_buf_filtered.shape)
        # print(occupy_points_index_filtered.shape)
        # 按照奖励统计
        self.accumulate_2d_rew(rew_buf_filtered, occupy_points_index_filtered, self.value_once, self.count_once)

        # 大约 50 episode 计算一次
        # # 按照成功次数统计
        # if self.common_step_counter % 2500 == 0:
        #     self._traversability_compute_count += 1
        #     # 计算单批次可通行得分
        #     total_once = self.num_succeed + self.num_fail
        #     self.traversability_score += torch.where(total_once > 0, self.num_succeed / total_once, torch.zeros_like(total_once))
        #     # 计算累计可通行得分
        #     once_mask = torch.where(total_once != 0, torch.ones_like(total_once), torch.zeros_like(total_once))
        #     self.num_total += once_mask
        #     self.traversability_score_mean = torch.where(self.num_total > 0, self.traversability_score / self.num_total, torch.zeros_like(self.num_total))
        #     # 成功失败buffer清零
        #     self.num_succeed.zero_()
        #     self.num_fail.zero_()
        #     print("compute traversability once!")
        # 按照奖励统计
        if self.common_step_counter % 2500 == 0:
            self._traversability_compute_count += 1
            # 计算单批次可通行得分（***注意归一化时不要带着未遍历区域）
                # 有效位置 mask
            scores = torch.where(self.count_once > 0, self.value_once / self.count_once, torch.zeros_like(self.count_once))
            valid_mask = self.count_once > 0
                # 先提取有效区域
            valid_scores = scores[valid_mask]
                # 计算有效区域的 min/max
            min_val = valid_scores.min()
            max_val = valid_scores.max()
                # 避免除 0
            eps = 1e-6
            norm_valid_scores = (valid_scores - min_val) / (max_val - min_val + eps)
                # 创建归一化后的 scores，初始化为 0
            normalized_scores = torch.zeros_like(scores)
                # 只把有效区域赋值回去
            normalized_scores[valid_mask] = norm_valid_scores
                # 用 normalized_scores 替换原 scores
            scores = normalized_scores
            #     # 归一化scores
            # eps = 1e-6
            # scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)
            self.traversability_score += scores
            # 计算累计可通行得分
            once_mask = torch.where(self.count_once != 0, torch.ones_like(self.count_once), torch.zeros_like(self.count_once))
            self.num_total += once_mask
            self.traversability_score_mean = torch.where(self.num_total > 0, self.traversability_score / self.num_total, torch.zeros_like(self.num_total))

            # 奖励累加值清零
            self.value_once.zero_()
            self.count_once.zero_()
            print("compute traversability once!")


        # # 可视化base投影
        # cube_geom = My_WireframeBoxGeometry(
        #     voxel_size*100, voxel_size*100, voxel_size*100,
        # )
        # occupy_points_index = occupy_points_index.cpu()
        # for n in range(occupy_points_index.shape[0]):
        #         i, j = int(occupy_points_index[n, 0].item()), int(occupy_points_index[n, 1].item())
        #         height = self.height_map[i, j]
        #         # 地图坐标转世界坐标
        #         x = i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
        #         y = j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
        #         # 将实际高度对齐到 voxel 栅格中心
        #         z_actual = height * self.cfg.terrain.vertical_scale# 中心在高度中心位置
        #         z = int(z_actual / voxel_size) * voxel_size + box_half
        #         cube_pose = gymapi.Transform()
        #         cube_pose.p = gymapi.Vec3(x, y, z)
        #         color = (1, 0, 0)
        #         # 创建 wireframe cube
        #         cube_geom.set_colors(color)
        #         # 画 cube
        #         gymutil.draw_lines(cube_geom, self.gym, self.viewer, self.envs[0], cube_pose)
    
    def accumulate_2d_rew(self, rew_buf, occupy_points_index, value_once, count_once):
        # value_once: (X, Y) → 扁平化为 1D 张量用于 scatter
        X, Y = value_once.shape
        flat_value = value_once.view(-1)
        flat_count = count_once.view(-1)
        # 提取 x 和 y 索引
        indices = occupy_points_index[:, :, :2].long()  # (4096, 15, 2)
        # 展平为 (4096*15, 2)
        indices_flat = indices.view(-1, 2)
        x = indices_flat[:, 0]
        y = indices_flat[:, 1]
        # 创建合法掩码（可选：防止负数或越界）
        mask = (x >= 0) & (x < X) & (y >= 0) & (y < Y)
        x = x[mask]
        y = y[mask]
        # 计算线性索引
        linear_idx = x * Y + y  # shape: [num_points_valid]
        # 将 rew_buf 扩展为 [4096, 15] → flatten 成 [4096*15]
        rew_flat = rew_buf.view(-1, 1).expand(-1, 15).reshape(-1)
        # 同样过滤掉对应的奖励
        rew_valid = rew_flat[mask]
        # scatter add 到网格中（flat_value 是 X*Y 长度）
        flat_value.scatter_add_(0, linear_idx, rew_valid)
        flat_count.scatter_add_(0, linear_idx, torch.ones_like(rew_valid))

    def accumulate_2d_hits(self, indices, map_2d):
        """
        indices: (N, 3) long tensor
        map_2d: (X, Y) tensor，原地累加
        """
        X, Y = map_2d.shape
        flat = map_2d.view(-1)
        indices = indices.view(-1, 3)
        linear_idx = indices[:, 0] * Y + indices[:, 1]
        # 使用布尔索引去除负值的元素
        mask = linear_idx >= 0
        linear_idx = linear_idx[mask].long()
        # print(f"Linear index min: {linear_idx.min()}, max: {linear_idx.max()}")
        flat.scatter_add_(0, linear_idx, torch.ones_like(linear_idx, dtype=flat.dtype))

    
    def export_traversability_voxel(self, filename="/home/ubuntu/Desktop/traversability_voxel.ply"):

        if self._traversability_drawn or self.height_samples is None:
            return

        start = time()
        voxel_size = 0.1
        box_half = voxel_size / 2.0

        rows, cols = self.traversability_score_mean.shape
        i, j = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')

        x = i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
        y = j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
        z_actual = self.height_map * self.cfg.terrain.vertical_scale
        z = (z_actual // voxel_size) * voxel_size + box_half
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(x.device)  # 保证 z 是 Tensor，且和 x 在同一设备上

        points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # (N, 3)
        scores = self.traversability_score_mean.reshape(-1)     # (N,)

        # 颜色范围：红（失败）到蓝（成功）
        # 红：(1, 0, 0), 蓝：(0, 0, 1)
        colors = torch.stack([
            1 - scores,           # R：高得分少红
            torch.zeros_like(scores),  # G
            scores                # B：得分越高越蓝
        ], dim=1)  # shape: (N, 3)
        
        # 当 total == 0 时，将其颜色设为灰色
        colors = torch.where(self.num_total.view(-1, 1) == 0, torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=colors.device), colors)

        # 构造 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy().astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy().astype(np.float32))

        # 保存为 .ply 文件
        o3d.io.write_point_cloud(filename, pcd)
        # o3d.io.write_point_cloud("smb://211.86.152.210/tiger-nas-readwrite/", pcd)
        print(f"[PointCloud] Exported to {filename}, {points.shape[0]} points.")
        self._traversability_drawn = True
        end = time()
        print(f"[VoxelDraw] Time: {end - start:.4f} s")
    

    def export_voxel_pointcloud(self, filename="/home/ubuntu/Desktop/voxel_surface.ply"):

        if self._voxel_drawn or self.height_samples is None:
            return

        start = time()
        voxel_size = 0.1
        box_half = voxel_size / 2.0

        # 高度图 tensor -> numpy
        height_map = self.height_samples.cpu().numpy()
        rows, cols = height_map.shape

        i_indices, j_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        x_coords = i_indices * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
        y_coords = j_indices * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
        z_actual = height_map * self.cfg.terrain.vertical_scale
        z_coords = (z_actual // voxel_size) * voxel_size + box_half

        # 点坐标
        points = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)

        # 归一化颜色
        max_h = np.max(height_map)
        min_h = np.min(height_map)
        range_h = max(max_h - min_h, 1e-5)
        norm = (height_map - min_h) / range_h
        r = norm
        g = 1.0 - 0.5 * norm
        b = np.zeros_like(norm)
        colors = np.stack([r, g, b], axis=-1).reshape(-1, 3)

        # 构造 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

        # 保存为 .ply 文件
        o3d.io.write_point_cloud(filename, pcd)
        # o3d.io.write_point_cloud("smb://211.86.152.210/tiger-nas-readwrite/", pcd)
        print(f"[PointCloud] Exported to {filename}, {points.shape[0]} points.")
        self._voxel_drawn = True
        end = time()
        print(f"[VoxelDraw] Time: {end - start:.4f} s")


    
    # def _get_octomap_style_color_vec(self, norm_heights):
    #     """
    #     批量计算颜色：从绿色 → 橙色，norm_heights 是 numpy 数组
    #     """
    #     r = norm_heights * 1.0
    #     g = 1.0 - 0.5 * norm_heights
    #     b = np.zeros_like(norm_heights)
    #     return np.stack([r, g, b], axis=-1)

    # def draw_voxel_terrain_surface(self):
    #     """
    #     在 Isaac Gym 中可视化 height_samples 的体素地形表面（只画每个栅格的顶部方块）
    #     """
    #     if self._voxel_drawn or self.height_samples is None:
    #         return
        
    #     start = time()

    #     self.gym.clear_lines(self.viewer)

    #     voxel_size = 0.1
    #     box_half = voxel_size / 2.0
    #     height_map = self.height_samples.cpu().numpy()
    #     rows, cols = height_map.shape

    #     # 计算地图坐标
    #     i_indices, j_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    #     x_coords = i_indices * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
    #     y_coords = j_indices * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
    #     z_actual = height_map * self.cfg.terrain.vertical_scale
    #     z_coords = (z_actual // voxel_size) * voxel_size + box_half

    #     # 归一化高度
    #     max_height = np.max(height_map)
    #     min_height = np.min(height_map)
    #     height_range = max(max_height - min_height, 1e-5)
    #     norm_heights = (height_map - min_height) / height_range

    #     # 批量计算颜色
    #     colors = self._get_octomap_style_color_vec(norm_heights)

    #     cube_pose = gymapi.Transform()
    #     cube_geom = My_WireframeBoxGeometry(
    #         voxel_size, voxel_size, voxel_size,
    #     )
    #     # 遍历绘制体素 cube（仅画顶部）
    #     for i in range(rows):
    #         for j in range(cols):
    #             cube_pose.p = gymapi.Vec3(
    #                 float(x_coords[i, j]),
    #                 float(y_coords[i, j]),
    #                 float(z_coords[i, j])
    #             )
    #             color = tuple(colors[i, j])  # (r, g, b)
    #             cube_geom.set_colors(color)
    #             gymutil.draw_lines(cube_geom, self.gym, self.viewer, self.envs[0], cube_pose)
    #     self._voxel_drawn = True
    #     end = time()
    #     print(f"[VoxelDraw] Time: {end - start:.4f} s")


    # def _get_octomap_style_color(self, norm_z):
    #     """
    #     颜色从绿色 → 橙色，norm_z ∈ [0, 1]
    #     """
    #     r = norm_z * 1.0       # R: 从 0 到 1
    #     g = 1.0 - 0.5 * norm_z # G: 从 1 到 0.5
    #     b = 0.0                # B: 始终为 0
    #     return (r, g, b)

    # def draw_voxel_terrain_surface(self):
    #     """
    #     在 Isaac Gym 中可视化 height_samples 的体素地形表面（只画每个栅格的顶部方块）
    #     """
    #     if self._voxel_drawn:
    #         return  # 避免重复执行
    #     if self.height_samples is None:
    #         return
    #     self.gym.clear_lines(self.viewer)
    #     start = time()
    #     # 体素尺寸
    #     voxel_size = 0.1
    #     box_half = voxel_size / 2.0
    #     # 取出 height_samples 并转换为 numpy
    #     height_map = self.height_samples.cpu().numpy()
    #     rows, cols = height_map.shape
    #     # 获取最大高度用于归一化（防止全是0或很小的浮点数）
    #     max_height = np.max(height_map)
    #     min_height = np.min(height_map)
    #     height_range = max(max_height - min_height, 1e-5)  # 防止除0
    #     cube_geom = My_WireframeBoxGeometry(
    #         voxel_size, voxel_size, voxel_size,
    #     )
    #     # 遍历每个格子，在其最高点画 cube
    #     for i in range(rows):
    #         for j in range(cols):
    #             height = height_map[i, j]
    #             # 地图坐标转世界坐标
    #             x = i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
    #             y = j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
    #             # 将实际高度对齐到 voxel 栅格中心
    #             z_actual = height * self.cfg.terrain.vertical_scale# 中心在高度中心位置
    #             z = int(z_actual / voxel_size) * voxel_size + box_half
    #             cube_pose = gymapi.Transform()
    #             cube_pose.p = gymapi.Vec3(x, y, z)
    #             # 归一化高度到 [0, 1]
    #             norm_h = (height - min_height) / height_range
    #             color = self._get_octomap_style_color(norm_h)
    #             # 创建 wireframe cube
    #             cube_geom.set_colors(color)
    #             # 画 cube
    #             gymutil.draw_lines(cube_geom, self.gym, self.viewer, self.envs[0], cube_pose)
    #     self._voxel_drawn = True  # 设置标志位
    #     end = time()
    #     print(f"[VoxelDraw] Time: {end - start:.4f} s")

    def destory_voxel_terrain_surface(self):
        """
        """
        if not self._voxel_drawn:
            return  # 避免重复执行
        if self.height_samples is None:
            return
        self.gym.clear_lines(self.viewer)
        self._voxel_drawn = False


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]
            
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
        
        self.proprioceptive_obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.proprioceptive_obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        if self.cfg.env.obs_with_lidar:
            self.obs_buf = torch.cat((self.proprioceptive_obs_buf, heights), dim=-1)
        else:
            self.obs_buf = self.proprioceptive_obs_buf
            
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def get_current_obs(self):
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]
            
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
            
        return current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]
            
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
        
        # return termination privileged obs
        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)

        # set y commands of high vel envs to zero
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #noise_vec[232:] = 0
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()


        # # 体素地图光线投影初始化
        #     # 设置 Raycast 参数
        # num_rays_x = 40
        # num_rays_y = 40
        # grid_size = 4.0
        # origin_z = 1.5
        #     # 定义射线起点和方向
        # x = torch.linspace(-grid_size/2, grid_size/2, num_rays_x)
        # y = torch.linspace(-grid_size/2, grid_size/2, num_rays_y)
        # xx, yy = torch.meshgrid(x, y, indexing="ij")
        # self.origins = torch.stack([
        #     xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), origin_z)
        # ], dim=-1)
        # self.dirs = torch.tensor([[0, 0, -1]]).repeat(self.origins.shape[0], 1)
        #     # 拓展为 batch: batch_size=1
        # self.origins = self.origins.unsqueeze(0).contiguous()
        # self.dirs = self.dirs.unsqueeze(0).contiguous()
        #     # GPU 分配
        # self.origins_tensor = gymtorch.unwrap_tensor(self.origins.cuda())
        # self.dirs_tensor = gymtorch.unwrap_tensor(self.dirs.cuda())
        #     # 初始化 Raycast 结果缓存
        # hits = torch.zeros_like(self.origins).cuda()


        # # 体素地图光线投影初始化
        #     # 射线参数
        # num_rays_x = 40
        # num_rays_y = 40
        # grid_size = 4.0
        # origin_z = 1.5
        # self.target_env_id = 0  # 我们只关心第一个环境的体素地图
        # # ==================== 构造射线 ====================
        # x = torch.linspace(-grid_size / 2, grid_size / 2, num_rays_x)
        # y = torch.linspace(-grid_size / 2, grid_size / 2, num_rays_y)
        # xx, yy = torch.meshgrid(x, y, indexing="ij")
        # xy_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [N, 2]
        # N = xy_points.shape[0]
        # self.origins = []
        # self.dirs_all = []
        # for i in range(self.num_envs):
        #     if i == self.target_env_id:
        #         # 目标环境使用真实射线
        #         z = torch.full((N,), origin_z)
        #         origin_i = torch.cat([xy_points, z.unsqueeze(-1)], dim=-1)  # [N, 3]
        #         dirs_i = torch.tensor([[0.0, 0.0, -1.0]]).repeat(N, 1)
        #     else:
        #         # 其他环境填充 dummy 射线（远处）
        #         origin_i = torch.full((N, 3), 1000.0)
        #         dirs_i = torch.tensor([[0.0, 0.0, -1.0]]).repeat(N, 1)
        #     self.origins.append(origin_i)
        #     self.dirs_all.append(dirs_i)
        # # 转为 tensor
        # self.origins = torch.stack(self.origins).cuda()       # [num_envs, N, 3]
        # self.dirs_all = torch.stack(self.dirs_all).cuda()     # [num_envs, N, 3]
        # self.origins_tensor = gymtorch.unwrap_tensor(self.origins.contiguous())
        # self.dirs_tensor = gymtorch.unwrap_tensor(self.dirs_all.contiguous())


        # 可通行性网格标签初始化
        self.height_map = self.height_samples.cpu().numpy()

        x = torch.tensor([-0.2, -0.1, 0, 0.1, 0.2], dtype=torch.float, device=self.device, requires_grad=False)
        y = torch.tensor([-0.1, 0, 0.1], dtype=torch.float, device=self.device, requires_grad=False)
        
        grid_x, grid_y = torch.meshgrid(x, y)
        num_points = grid_x.numel()
        self.base_occupy_points = torch.zeros(num_points, 3, device=self.device, requires_grad=False)
        self.base_occupy_points[:, 0] = grid_x.flatten()
        self.base_occupy_points[:, 1] = grid_y.flatten()

        rows, cols = self.height_samples.shape
        self.num_succeed = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.num_fail = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.num_total = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.value_once = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.count_once = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.traversability_score = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.traversability_score_mean = torch.zeros((rows, cols), dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_count = torch.full((self.num_envs,), 15, dtype=torch.int, device=self.device, requires_grad=False)



        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        print(self.dof_names)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
