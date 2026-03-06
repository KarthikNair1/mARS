import os
import torch.backends.cudnn as cudnn
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from datetime import datetime 
import pdb
import yaml
import sys
import glob
import shutil
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
from tqdm import tqdm

#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, GuidedGradCam
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz

import h5py
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


if __name__ == '__main__':
    color_dict = {}
    np.random.seed(2025)
    for i in range(255):
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        color_dict[i] = color
    color_dict[0] = (0, 0, 0)  # background as black

    def load_h5_arr(file):
        h5 = h5py.File(file, 'r')
        arr = h5['image'][:]
        h5.close()
        return arr

    # argparse to accept start and end index
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_nodes', type=int)
    parser.add_argument('--node_index', type=int)
    #parser.add_argument('--start', type=int, default=0, help='Start index')
    #parser.add_argument('--end', type=int, default=10000, help='End index')
    args = parser.parse_args()


    tasks = ['total', 'vertebrae_body', 'appendicular_bones']

    rois = [
    "sacrum",
    "vertebrae_S1",	"vertebrae_L5","vertebrae_L4","vertebrae_L3","vertebrae_L2","vertebrae_L1",
    "vertebrae_T12","vertebrae_T11","vertebrae_T10","vertebrae_T9","vertebrae_T8","vertebrae_T7",
    "vertebrae_T6","vertebrae_T5","vertebrae_T4","vertebrae_T3","vertebrae_T2","vertebrae_T1","vertebrae_C7","vertebrae_C6","vertebrae_C5","vertebrae_C4","vertebrae_C3",
    "vertebrae_C2","vertebrae_C1",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",	
    "hip_left",
    "hip_right",
    "skull",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12"
    ]

    def get_seg_for_row(row, file):
        sz_x = row['pixel_size_1'].values[0]
        sz_y = row['pixel_size_2'].values[0]
        sz_z = row['pixel_size_3'].values[0]
        if sz_x - 0.0 < 1e-5 or sz_y - 0.0 < 1e-5 or sz_z - 0.0 < 1e-5:
            print(f'Invalid pixel size for {file}, skipping...', (sz_x, sz_y, sz_z))
            return None
        
        affine_arr = np.array([ [sz_x, 0, 0, 0],
                                [0, sz_y, 0, 0],
                                [0, 0, sz_z, 0],
                                [0, 0, 0, 1]])

        arr = load_h5_arr(file)

        new_image = nib.Nifti1Image(arr, affine=affine_arr)

        # union all tasks

        unioned_segmentation = np.zeros(arr.shape)
        for task in tasks:
            if task == 'total':
                segmented = totalsegmentator(new_image, task = task, roi_subset = rois)
            else:
                segmented = totalsegmentator(new_image, task = task)
            
            unioned_segmentation = np.maximum(unioned_segmentation, segmented.get_fdata() > 0)

        return unioned_segmentation
    
    full_data_csv = pd.read_csv('/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/data_scan_ct_filtered_for_segmentation_resampled_to_pet.csv')
    # subset full_data_csv based on args
    start_idx = int((full_data_csv.shape[0] / args.total_nodes) * args.node_index)
    end_idx = int((full_data_csv.shape[0] / args.total_nodes) * (args.node_index + 1))
    full_data_csv = full_data_csv.iloc[start_idx:min(end_idx, full_data_csv.shape[0])].reset_index(drop=True)
    print('Processing indices from {} to {}'.format(start_idx, end_idx))

    #rt = '/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_h5'
    rt = '/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_h5_aligned_ct'

    # with tqdm
    for index, row in tqdm(full_data_csv.iterrows(), total=full_data_csv.shape[0]):
        file = os.path.join(rt, f"{row['filepath']}/{row['filename']}")
        search_fullpath = file.split(f'{rt}/')[1]
        row_df = full_data_csv[full_data_csv['fullfile'] == search_fullpath]

        output_dir = f"/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/ct_bone_segmentations_2/{row['filepath']}"
        output_file = os.path.join(output_dir, f"{row['filename'].replace('.h5', '_bone_segmentation.h5')}")
        if os.path.exists(output_file):
            print(f'Bone segmentation already exists for {file}, skipping...')
            continue

        os.makedirs(output_dir, exist_ok=True)

        unioned_segmentation = get_seg_for_row(row_df, file)
        if unioned_segmentation is None:
            print(f'No segmentation generated for {file}')
            continue
        
        # write to h5
        
        with h5py.File(output_file, 'w') as h5:
            h5.create_dataset('segmentation', data=unioned_segmentation.astype(np.uint8))

        print(f'Saved bone segmentation to {output_file}')