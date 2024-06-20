import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..dense_heads.occ_head_plugin import IntersectionOverUnion, PanopticMetric
from ..dense_heads.planning_head_plugin import PlanningMetric
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC

from scripts_for_ad.data_process import classify_cam_files, get_cam_data

import mmcv
import numpy as np
import pycocotools.mask as mask_util
import json

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()

    # Occ eval init
    eval_occ = hasattr(model.module, 'with_occ_head') \
                and model.module.with_occ_head
    if eval_occ:
        # 30mx30m, 100mx100m at 50cm resolution
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            iou_metrics[key] = IntersectionOverUnion(n_classes).cuda()
        panoptic_metrics = {}
        for key in EVALUATION_RANGES.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
    
    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head
    #for carla
    eval_planning = False
    eval_occ = False

    if eval_planning:
        planning_metrics = PlanningMetric().cuda()
        
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    for i, data in enumerate(data_loader):
        #breakpoint()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            #write resule to txt file
            if i == 0:
                #print("result: ", result[0]['bbox_results'])
                pass

            # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']
                planning_metrics(pred_sdc_traj[:, :6, :2], sdc_planning[0][0,:, :6, :2], sdc_planning_mask[0][0,:, :6, :2], segmentation[0][:, [1,2,3,4,5,6]])
                #print("planning output: ", pred_sdc_traj[:, :6, :2])    
            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    num_occ += 1
                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        iou_metrics[key](result[0]['occ']['seg_out'][..., limits, limits].contiguous(),
                                        result[0]['occ']['seg_gt'][..., limits, limits].contiguous())
                        panoptic_metrics[key](result[0]['occ']['ins_seg_out'][..., limits, limits].contiguous().detach(),
                                                result[0]['occ']['ins_seg_gt'][..., limits, limits].contiguous())

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:
                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid', 'seg_out', 'ins_seg_out']:
                    if k in result[0]['occ']:
                        result[0]['occ'][k] = result[0]['occ'][k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()
                        print(f"Detached {k} from GPU", result[0]['pts_bbox'][k])

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if eval_planning:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results


def custom_multi_gpu_test_carla(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()

    # Occ eval init
    eval_occ = hasattr(model.module, 'with_occ_head') \
                and model.module.with_occ_head
    if eval_occ:
        # 30mx30m, 100mx100m at 50cm resolution
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            iou_metrics[key] = IntersectionOverUnion(n_classes).cuda()
        panoptic_metrics = {}
        for key in EVALUATION_RANGES.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
    
    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head
    #for carla
    eval_planning = False
    eval_occ = False

    if eval_planning:
        planning_metrics = PlanningMetric().cuda()
        
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0


    data_format = data_loader.dataset[0]
    datas_all = []


    canbus_path = 'data/carla-gen_data-1agent-20242303/39/0039_pose.json'
    ego_pose_path = 'data/carla-gen_data-1agent-20242303/39/ego_pose_0039.json'
    calibration_sensor_path = 'data/carla-gen_data-1agent-20242303/calibration_sensor.json'
    command_and_time_path = 'data/carla-gen_data-1agent-20242303/39/0039_command.json'
    cam_data_path = 'data/carla-gen_data-1agent-20242303/39'
    #read json file
    with open(canbus_path) as f:
        canbus_datas = json.load(f)

    with open(ego_pose_path) as f:
        ego_pose = json.load(f)

    with open(calibration_sensor_path) as f:
        calibration_sensor = json.load(f)

    with open(command_and_time_path) as f:
        command_and_time = json.load(f)

    #front_cam= os.listdir('data/06122152_1agent_res1600x928/39/39_front')
    file_names = classify_cam_files(cam_data_path) 
    cam_tensor = get_cam_data(file_names) #[N, 1, M, C, H, W]
    
    
    #breakpoint()
    lidar2img_all=[]
    for cam_id, cam_data in enumerate(calibration_sensor):
        #lidar2cam_rt = np.eye(4)
        
        lidar2cam_rt= np.array(cam_data['lidar2cam'])
        viewpad = np.eye(4)
        intrinsic = np.array(cam_data['camera_intrinsic'])
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        lidar2img_rt = (viewpad @ lidar2cam_rt)
        lidar2img_all.append(lidar2img_rt)

    for canbus_id, canbus_data in enumerate(canbus_datas):
        can_bus = []

        #data_carla = data_format.copy()
        #breakpoint()
        pos = ego_pose[canbus_id]['translation']
        rot = ego_pose[canbus_id]['rotation']      
        acc = canbus_data['accel']
        vel = canbus_data['vel']
        angle = canbus_data['rotation_rate']
        can_bus.extend(pos)
        can_bus.extend(rot)
        can_bus.extend(acc)
        can_bus.extend(vel)
        can_bus.extend(angle)
        can_bus.extend([0., 0.])

        #TODO: should check
        rot = Quaternion(rot)
        patch_angle = quaternion_yaw(rot) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        can_bus = np.array(can_bus)
        
        command = command_and_time[canbus_id]['command']
        timestamp = command_and_time[canbus_id]['utime']

        info = dict(
                filename = file_names[canbus_id],
                ori_shape = data_format['img_metas'][0].data['ori_shape'],
                img_shape = data_format['img_metas'][0].data['img_shape'],
                lidar2img = lidar2img_all,
                pad_shape = data_format['img_metas'][0].data['pad_shape'],
                scale_factor = data_format['img_metas'][0].data['scale_factor'],
                flip = data_format['img_metas'][0].data['flip'],
                pcd_horizontal_flip = data_format['img_metas'][0].data['pcd_horizontal_flip'],
                pcd_vertical_flip = data_format['img_metas'][0].data['pcd_vertical_flip'],
                box_mode_3d = data_format['img_metas'][0].data['box_mode_3d'],
                box_type_3d = data_format['img_metas'][0].data['box_type_3d'],
                img_norm_cfg = {}, #TODO:should check
                sample_idx = str(canbus_id),
                pcd_scale_factor = data_format['img_metas'][0].data['pcd_scale_factor'],
                scene_token = 'calar_39',
                can_bus = can_bus,
        )

        img_metas=DC([[info]])

        data_carla = dict(
            img_metas=[img_metas],
            img=[DC([cam_tensor[canbus_id].to(torch.float32)])],
            timestamp = [torch.tensor(timestamp).to(torch.float64).unsqueeze(0)],
            command = [torch.tensor(command).unsqueeze(0)],
        )

        
        # data_carla['img_metas'][0].data['filename'] = file_names[canbus_id]

        # data_carla['img_metas'][0].data['can_bus'] = can_bus  
        # #TODO：check
        # data_carla['img_metas'][0].data['lidar2img'] = lidar2img_all
        

        # #breakpoint()
        # data_carla['img'][0]= DC(cam_tensor[canbus_id].to(torch.float32))

        # data_carla['timestamp'][0] = timestamp
        # data_carla['command'][0] = command

        # data_carla['img_metas'][0].data['img_norm_cfg'] = {}

        # data_carla['img_metas'][0].data['sample_idx']= canbus_id   
        # data_carla['img_metas'][0].data['scene_token']= 'carla_39'

        #breakpoint()
        #breakpoint()
        datas_all.append(data_carla)

    for data_test in data_loader:
        data = data_test
        break
    #for i, data in enumerate(data_loader):
    for i in range(len(datas_all)):

        #change data to carla
        # data['img'][0].data[0][0,0,0,0] =0
        # data['img_metas'][0].data[0][0]['filename'] =[]
        data['img_metas'][0].data[0][0]['img_norm_cfg'] = {}
        # data['img_metas'][0].data[0][0]['std'] = None
        # breakpoint()
        # data['img_metas'][0].data['img_norm_cfg'] = {}
        lidar2img = data['img_metas'][0].data[0][0]['lidar2img'].copy()
        data['img_metas'][0].data[0] = datas_all[i]['img_metas'][0].data[0]
        
        #data['img_metas'][0].data[0][0]['lidar2img'] = lidar2img
        #breakpoint()
        #data = datas_all[i]
        #data['img_metas'][0].data[0][0]['can_bus'] = datas_all[i]['img_metas'][0].data[0][0]['can_bus']
        #easily modify data 
        data['img'][0].data[0] = datas_all[i]['img'][0].data[0]
        #data['img_metas'][0].data[0] = datas_all[i]['img_metas'][0].data[0]
        data['timestamp'][0] = datas_all[i]['timestamp'][0]
        data['command'][0] = datas_all[i]['command'][0]




        breakpoint()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            #write resule to txt file
            if i == 0:
                #print("result: ", result[0]['bbox_results'])
                pass

            # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']
                planning_metrics(pred_sdc_traj[:, :6, :2], sdc_planning[0][0,:, :6, :2], sdc_planning_mask[0][0,:, :6, :2], segmentation[0][:, [1,2,3,4,5,6]])
                #print("planning output: ", pred_sdc_traj[:, :6, :2])    
            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    num_occ += 1
                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        iou_metrics[key](result[0]['occ']['seg_out'][..., limits, limits].contiguous(),
                                        result[0]['occ']['seg_gt'][..., limits, limits].contiguous())
                        panoptic_metrics[key](result[0]['occ']['ins_seg_out'][..., limits, limits].contiguous().detach(),
                                                result[0]['occ']['ins_seg_gt'][..., limits, limits].contiguous())

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:
                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid', 'seg_out', 'ins_seg_out']:
                    if k in result[0]['occ']:
                        result[0]['occ'][k] = result[0]['occ'][k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()
                        print(f"Detached {k} from GPU", result[0]['pts_bbox'][k])

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if eval_planning:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)