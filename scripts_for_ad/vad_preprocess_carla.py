import os
import math
import copy
import pickle
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import transform_matrix

from vad_preprocess import quart_to_rpy

import json

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

ego_width, ego_length = 1.85, 4.084


def _get_can_bus_info(canbus_data):
    can_bus = []

    can_bus.extend(canbus_data['pos'])
    can_bus.extend(canbus_data['orientation'])
    can_bus.extend(canbus_data['accel'])
    can_bus.extend(canbus_data['rotation_rate'])
    can_bus.extend(canbus_data['vel'])
    can_bus.extend([0., 0.])

    return can_bus




def get_global_sensor_pose(ego2global, lidar2ego, inverse=False):
    lidar2ego_rt = lidar2ego['lidar2ego']
    ego2global_rt = transform_matrix(ego2global['translation'], Quaternion(ego2global['rotation']), inverse=inverse)

    if inverse:
        return np.dot(lidar2ego_rt, ego2global_rt) #TODO:should check the order
    else:
        return np.dot(np.linalg.inv(ego2global_rt), np.linalg.inv(lidar2ego_rt))



def fill_carla_infos(carla_dir, scene_name, his_ts=4, fut_ts=6):


    lidar2ego_path = os.path.join(carla_dir, scene_name, 'lidar2local.json')
    #calibration_path = os.path.join(carla_dir, scene_name, 'calibration.json')
    ego_pose_path = os.path.join(carla_dir, scene_name, 'ego_pose.json')
    canbus_path = os.path.join(carla_dir, scene_name, 'pose.json')

    steer_path = os.path.join(carla_dir, scene_name, 'steer_angle_feedback.json')

    with open(canbus_path) as f:
        canbus_datas = json.load(f)
    
    with open(ego_pose_path) as f:
        ego_pose_datas = json.load(f)

    with open(lidar2ego_path) as f:
        lidar2ego_data = json.load(f)
        lidar2ego = np.array(lidar2ego_data['lidar2ego'])

    with open(steer_path) as f:
        steer_datas = json.load(f)

    assert len(canbus_datas) == len(ego_pose_datas), 'canbus and ego pose data length not equal.'
    print(f'update {len(ego_pose_datas)} samples, start to add ego state and hist/fut traj/traj_diff .')

    for frame_id, canbus_data in enumerate(canbus_datas):


        ego_his_trajs = np.zeros((his_ts+1, 3))
        ego_his_trajs_diff = np.zeros((his_ts+1, 3))  
        sample_cur = frame_id
        
        # get the history trajectory of the ego vehicle in the gloabl coordinatre
        for i in range(his_ts, -1, -1):

            if sample_cur is not None:
                pose_mat = get_global_sensor_pose(ego_pose_datas[sample_cur], lidar2ego_data, inverse=False)
                ego_his_trajs[i] = pose_mat[:3, 3]

                next = sample_cur + 1
                if next >=0 and next < len(ego_pose_datas):
                    pose_mat_next = get_global_sensor_pose(ego_pose_datas[next], lidar2ego_data, inverse=False)
                    ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - pose_mat[:3, 3]
                


                prev = sample_cur - 1
                has_prev = prev >= 0 and prev < len(ego_pose_datas)

                sample_cur = prev if has_prev else None

            else:
                ego_his_trajs[i] = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
                ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]



        # global to ego at lcf
        ego_his_trajs = ego_his_trajs - np.array(ego_pose_datas[frame_id]['translation'])
        rot_mat = Quaternion(ego_pose_datas[frame_id]['rotation']).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        # ego to lidar at lcf
        #breakpoint()
        ego_his_trajs = ego_his_trajs - lidar2ego[:3,3]
        rot_mat = lidar2ego[:3,:3].T
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        ego_his_diff = ego_his_trajs[1:] - ego_his_trajs[:-1]
    

        ego_fut_trajs = np.zeros((fut_ts+1, 3))
        ego_fut_masks = np.zeros((fut_ts+1))
        sample_cur = frame_id

        for i in range(fut_ts+1):
            pose_mat = get_global_sensor_pose(ego_pose_datas[sample_cur], lidar2ego_data, inverse=False)
            ego_fut_trajs[i] = pose_mat[:3, 3]
            ego_fut_masks[i] = 1

            next = sample_cur + 1
            if next>=len(ego_pose_datas):
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    break
            else:
                sample_cur = next

        # global to ego at lcf
        ego_fut_trajs = ego_fut_trajs - np.array(ego_pose_datas[frame_id]['translation'])
        rot_mat = Quaternion(ego_pose_datas[frame_id]['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # ego to lidar at lcf
        ego_fut_trajs = ego_fut_trajs - lidar2ego[:3,3]
        rot_mat = lidar2ego[:3,:3].T
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

        if ego_fut_trajs[-1][0] >= 2:
            command = np.array([1, 0, 0])  # Turn Right
        elif ego_fut_trajs[-1][0] <= -2:
            command = np.array([0, 1, 0])  # Turn Left
        else:
            command = np.array([0, 0, 1])  # Go Straight

        ego_fut_diff = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

        ego_lcf_feat = np.zeros(9)

        _, _, ego_yaw = quart_to_rpy(ego_pose_datas[frame_id]['rotation'])
        ego_pos = np.array(ego_pose_datas[frame_id]['translation'])

        prev_frame_id = frame_id - 1
        pose_record_prev = prev_frame_id >= 0

        next_frame_id = frame_id + 1
        pose_record_next = next_frame_id < len(ego_pose_datas)
        if pose_record_prev:
            _, _, ego_yaw_prev = quart_to_rpy(ego_pose_datas[prev_frame_id]['rotation'])
            ego_pos_prev = np.array(ego_pose_datas[prev_frame_id]['translation'])

        if pose_record_next:
            _, _, ego_yaw_next = quart_to_rpy(ego_pose_datas[next_frame_id]['rotation'])
            ego_pos_next = np.array(ego_pose_datas[next_frame_id]['translation'])

        assert (pose_record_prev ) or (pose_record_next ), 'prev token and next token all empty'
        
        if pose_record_prev:
            ego_w = (ego_yaw - ego_yaw_prev) / 0.5
            ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
            ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
        else:
            ego_w = (ego_yaw_next - ego_yaw) / 0.5
            ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
            ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

        try:
            v0 = canbus_data['vel'][0]
            steering = steer_datas[frame_id]['value']
            Kappa = 2 * steering / 2.588 #TODO: check the steering ratio
        
        except:
            delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
            delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
            v0 = np.sqrt(delta_x**2 + delta_y**2)
            Kappa = 0



        can_bus =  _get_can_bus_info(canbus_data)

        ego_lcf_feat[:2] = np.array([ego_vx, ego_vy]) #can_bus[13:15]
        ego_lcf_feat[2:4] = can_bus[7:9]
        ego_lcf_feat[4] = ego_w #can_bus[12]
        ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
        ego_lcf_feat[7] = v0
        ego_lcf_feat[8] = Kappa

        info_align_ad = {
            'goal': command.astype(np.float32),
            'ego_states': ego_lcf_feat.astype(np.float32), 
            'ego_hist_traj': ego_his_trajs[:, :2].astype(np.float32), 
            'ego_hist_traj_diff': ego_his_diff[:, :2].astype(np.float32),  
            'ego_fut_traj': ego_fut_trajs[:, :2].astype(np.float32),  
            'ego_fut_traj_diff': ego_fut_diff[:, :2].astype(np.float32), 
            } 

        token = str(frame_id)
        info_name = token + '.pkl'
        info_path = os.path.join('extra_data/val', info_name)

        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        data.update(info_align_ad)

        with open(info_path, 'wb') as f:
            pickle.dump(data, f)
    
        print(f'update {info_name}, adding ego state and hist/fut traj/traj_diff .')
        
    print(f'\n <<<<<<<<<update {frame_id} samples, finsihed.<<<<<<<<<')

if __name__ == "__main__":
   
   carla_dir = 'data/06202142-3_agent'
   scene_name = '55'

   fill_carla_infos(carla_dir, scene_name, his_ts=4, fut_ts=6)