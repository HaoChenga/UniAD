import glob
import os
import cv2
import torch
import mmcv
import numpy as np

def get_sorted_cam_files(file_path):
    """
    Process the camera data.
    get file name and sort them by the frame number.

    return the sorted file list.
    """
    cam_files = glob.glob(file_path + '/*.png')
    cam_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return cam_files


def classify_cam_files(file_path):
    """
    Classify the camera data by the time.
    input view:[front, front_right, front_left, back, back_right, back_left]
    """

    cam_view = ['front', 'front_right', 'front_left', 'back', 'back_left', 'back_right']
    cam_files = []
    for view in cam_view:
        cam_file = os.path.join(file_path, view)
        cam_file = get_sorted_cam_files(cam_file)
        cam_files.append(cam_file)

    #classify the camera data by the time.
    cam_files = [list(x) for x in zip(*cam_files)]
    return cam_files


def get_cam_data(cam_files):
    """
    Get the camera data from the file.
    input: list of camera file path.
    output: list of camera data(tensor). [N, 1, M, C, H, W]
    """
    img_norm_cfg = dict(mean=np.array([103.530, 116.280, 123.675]), std=np.array([1.0, 1.0, 1.0]), to_rgb=False)
    cam_data = []

    for cam_file_syc in cam_files:
        cam_data_syc = []
        for cam_file in cam_file_syc:
            cam_data_syc.append(mmcv.imread(cam_file, flag='unchanged')[...,:3])
            #to tensor

        cam_data_syc = np.stack(cam_data_syc, axis=-1)
        cam_data_syc = cam_data_syc.astype(np.float32)
        cam_data_syc_ = [cam_data_syc[..., i] for i in range(cam_data_syc.shape[-1])]
        padded_img = [mmcv.impad_to_multiple(
                img, 32, 0) for img in cam_data_syc_]

        padded_img_norm = [mmcv.imnormalize(img, img_norm_cfg['mean'], img_norm_cfg['std'], img_norm_cfg['to_rgb']) for img in padded_img]

        #cam_data_syc = torch.stack(cam_data_syc)
        padded_img_norm = torch.from_numpy(np.stack(padded_img_norm).transpose(0,3,1,2))
        padded_img_norm = padded_img_norm.unsqueeze(0)    
            #breakpoint()
        cam_data.append(padded_img_norm)
    cam_data = torch.stack(cam_data) 
    return cam_data