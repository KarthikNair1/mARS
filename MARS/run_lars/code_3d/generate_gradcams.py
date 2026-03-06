import torch
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, GuidedGradCam, Saliency, DeepLift, GuidedBackprop, InputXGradient
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings
from medcam import medcam

import os
import torch.backends.cudnn as cudnn
import torch
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
import imageio

import utils
from dataset import get_datasets_3d, dataset_3d, dataset_3d_center
from utils import model_prediction, my_model
def load_config():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def set_defaults_config():
    default_dict = {
        'postfilter_fwhm_xy_mm': None,
        'normalize_mean': None,
        'normalize_std': None,
        'split': 'val',
        'layer_idx': 4,
        'ct_mode': None,
        'ct_bone_seg_dilation_erosion_pixels': None,
        'ct_bone_seg_dilation_alone_pixels': None,
        'binarization_threshold': None
    }

    for key, value in default_dict.items():
        if key not in args:
            args[key] = value

def load_merged_dfs(binarization_threshold = 0.5):
    # load model probabilities for each scan
    dfs_list = []
    for i in range(10):
        dfr = pd.read_csv(os.path.join(args["output"], args["dataset_type"], f'pred_{args["dataset_type"]}_{i}.csv'))
        col_list = ['filepath', 'PET_jco', 'PET_imwg', 'probs']
        if 'mipview' in dfr.columns: col_list.append('mipview')
        dfr = dfr[col_list]
        dfr['model_number'] = i
        dfs_list.append(dfr)
    model_probabilities = pd.concat(dfs_list)
    model_probabilities.rename({'filepath': 'accession'}, axis=1, inplace=True)
    
    # next, load data for each scan and merge
    split = pd.read_pickle('/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl')
    df = model_probabilities[model_probabilities['accession'].isin(split[args["split"]])]
    binarization_threshold = binarization_threshold # for now

    uses_mipview = 'mipview' in model_probabilities.columns
    col_list = ['accession', 'probs', 'model_number']
    if uses_mipview: col_list.append('mipview')
    df = df[col_list]

    filt_col = ['accession']
    if uses_mipview: filt_col.append('mipview')
    df = df.groupby(filt_col).agg({'probs': 'mean'}).reset_index()

    target = pd.read_excel('/gpfs/data/mayerhoeferlab/projects/myeloma/curated_status/Myeloma_master_ref_15_08_2025.xlsx')
    target = target.rename(columns={'ACC':'accession','PET-jco':'PET_jco','PET-imwg':'PET_imwg'})
    target = target[['accession', 'PET_jco', 'PET_imwg']]

    # merge to get labels
    col_list = ['accession', 'probs', 'PET_jco', 'PET_imwg']
    if uses_mipview: col_list.append('mipview')
    merged = pd.merge(df, target[['accession', 'PET_jco', 'PET_imwg']] , on='accession', how='left')
    merged = merged[col_list]

    # exclude cases where there was no label available
    merged = merged.dropna(subset = ['PET_jco'], axis='index')

    # also exclude cases where the label is 2 (just for now)
    merged = merged[merged['PET_jco'] != 2]

    merged_before_drop_mipview = merged.copy()
    # average probs across mipview
    merged = merged.groupby(['accession']).agg({'probs': ['mean', 'max', 'min'], 'PET_jco': 'first', 'PET_imwg': 'first'}).reset_index()
    merged.columns = [' '.join(col).strip() for col in merged.columns.values]
    merged['mip_difference'] = merged['probs max'] - merged['probs min']
    merged = merged.rename({'probs mean': 'probs', 'PET_jco first': 'PET_jco', 'PET_imwg first': 'PET_imwg'}, axis=1)
    merged = merged.drop(['probs max', 'probs min'], axis=1)
    merged['difference'] = abs(merged['probs'] - merged['PET_jco'])

    merged['binarized_prediction']  = (merged['probs'] >= binarization_threshold).astype(int)
    merged = merged[['accession', 'probs', 'difference', 'mip_difference', 'binarized_prediction', 'PET_jco', 'PET_imwg']]
    merged = merged.sort_values(by=['accession'])
    
    return (model_probabilities, merged, merged_before_drop_mipview)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (1, '#0000ff')], N=256)

cmap_2 = LinearSegmentedColormap.from_list('',
                                                 [(0, "#ff0000"),
                                                  (0.5, "#FFFFFF"),
                                                  (1, '#0000ff')], N=256)


def dataset_tensor_to_rgb_img(tensor):
    # tensor should have shape (C, 250, 250, 300)
    nparray = torch.flatten(tensor, start_dim=0, end_dim=1).numpy() # (C*250, 250, 300)
    nparray = np.stack((nparray,)*3, axis=3) # (C*250, 250, 300, 3)
    nparray = (nparray - nparray.min()) / (nparray.max() - nparray.min())
    nparray = nparray * 255
    nparray = nparray.astype(np.uint8)
    return nparray

def calculate_interpretability_for_study_3D(merged_df, accession_id, label_number, type='IntegratedGradients', layer_to_iterate = None, device='cuda', model_idx = None):
    pred_label_idx = label_number
    rr = dset.df[(dset.df['accession'] == accession_id)]
    idx = rr.index[0]
    assert len(rr) == 1, "accession_id not found or not unique"
    rr = dset[idx][0] # get the tensor
    
    rgb_img = dataset_tensor_to_rgb_img(rr[0:1]) # convert to rgb image
    ct_img = dataset_tensor_to_rgb_img(rr[1:2])

    rr = rr[0:1,:,:,:] # only use channel 0 for interp, channel 1 (CT) is just for visualization
    input_shape_per_batch = rr.shape

    predprob = merged_df[(merged_df['accession'] == accession_id)]['probs'].values[0]
    if model_idx is not None:
        predprob = model_probabilities[(model_probabilities['accession'] == accession_id) & (model_probabilities['model_number'].isin(model_idx))]['probs']
        predprob = predprob.mean()
        #assert len(predprob) == 1, "Prediction probability not found or not unique"
    inp_tensor = rr.unsqueeze(0).to(device)
    attribution_list = []
    inp_tensor.requires_grad_()

    for model_this_idx in model_idx:
        model_to_use = models_list[model_this_idx].to(device).eval()
        target_layer = model_to_use.features[layer_to_iterate][2].conv2
        if type == 'medcam':
            model_to_use = medcam.inject(model_to_use, output_dir = '/gpfs/data/mayerhoeferlab/projects/myeloma/results/3D_all_seeds_prediction/val/interpretability/special_medcam', layer='features.5.2.conv2', return_attention=True, backend='ggcam')
            # inference on inp_tensor
            _, attributions = model_to_use(inp_tensor)
        
        if type == 'pytorch-gradcam':
            targets = [ClassifierOutputTarget(label_number)]
            
            with GradCAM(model=model_to_use, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=inp_tensor, targets=targets)

            attributions = np.reshape(grayscale_cam, (input_shape_per_batch[0] * input_shape_per_batch[1], input_shape_per_batch[2], input_shape_per_batch[3])) # remove channel dimension by flattening, for visualization against rgb_img
            # convert back to tensor
            attributions = torch.tensor(attributions, device=device)
        
        elif type == 'IntegratedGradients':
            obj = IntegratedGradients(model_to_use)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx, n_steps=200)
        elif type == 'LayerGradCam':
            obj = LayerGradCam(model_to_use, target_layer)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx)
            attributions = LayerAttribution.interpolate(attributions, inp_tensor.shape[2:])
        elif type == 'GuidedGradCam':
            obj = GuidedGradCam(model_to_use, target_layer)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx)
        elif type == 'Occlusion':
            obj = Occlusion(model_to_use)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx,
                                        sliding_window_shapes=(1, 8, 8, 8), 
                                        strides=(1, 8, 8, 8), 
                                        baselines=0)
        elif type=='Saliency':
            obj = Saliency(model_to_use)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx)
        elif type=='InputXGradient':
            obj = InputXGradient(model_to_use)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx)
        elif type=='GuidedBackprop':
            obj = GuidedBackprop(model_to_use)
            attributions = obj.attribute(inp_tensor, target=pred_label_idx)
        
        attribution_list.append(attributions)

    attributions = torch.mean(torch.stack(attribution_list), dim=0)
    attributions = attributions.cpu().detach().squeeze().view(input_shape_per_batch)
    if attributions.shape[0] != 1:
        # flatten dimension to match dims of rgb_img for visualization
        attributions = torch.flatten(attributions, start_dim=0, end_dim=1)
        attributions = attributions.unsqueeze(0)
    attributions = attributions.numpy()
    
    return (np.uint8(255 - rgb_img), np.uint8(ct_img), attributions, predprob)

def visualize_3d_vol_and_attr(vol, ct_vol, attribution, output_file_path):
    attribution = attribution.squeeze()
    print(vol.shape)
    print(ct_vol.shape)
    print('===')
    # vol has shape (C*D, H, W, 3)
    # ct_vol has shape (C*D, H, W, 3)
    # attribution has shape (1, C*D, H, W)

    #attribution_transf = np.repeat(attribution.squeeze(axis=0)[:,:,:,np.newaxis], repeats=3, axis=3)
    #attribution_transf = (attribution_transf - attribution_transf.min()) / (attribution_transf.max() - attribution_transf.min())
    #attribution_transf = (attribution_transf * 255).astype(np.uint8)

    vol = vol.astype(np.uint8)
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min())

    ct_vol = ct_vol.astype(np.uint8)

    # crop vol
    min_d1 = np.where(vol.min(axis = (1,2,3)) < 255)[0].min()
    max_d1 = np.where(vol.min(axis = (1,2,3)) < 255)[0].max() + 1
    min_d2 = np.where(vol.min(axis = (0,2,3)) < 255)[0].min()
    max_d2 = np.where(vol.min(axis = (0,2,3)) < 255)[0].max() + 1
    min_d3 = np.where(vol.min(axis = (0,1,3)) < 255)[0].min()
    max_d3 = np.where(vol.min(axis = (0,1,3)) < 255)[0].max() + 1

    vol = vol[min_d1:max_d1, min_d2:max_d2, min_d3:max_d3]
    vol_norm = vol_norm[min_d1:max_d1, min_d2:max_d2, min_d3:max_d3]
    ct_vol = ct_vol[min_d1:max_d1, min_d2:max_d2, min_d3:max_d3]
    attribution_transf = attribution
    attribution_transf = attribution_transf[min_d1:max_d1, min_d2:max_d2, min_d3:max_d3]

    # output as side-by side GIF, where left half is volume and right half is attribution
    # output as gif

    # scale attribution to be 0...1
    print(attribution_transf.min(), attribution_transf.max())
    attribution_transf = (attribution_transf - attribution_transf.min()) / (attribution_transf.max() - attribution_transf.min())
    side_by_side_frames = []
    for i in range(vol.shape[0]):
        outp = show_cam_on_image(vol_norm[i,:,:,:], attribution_transf[i,:,:], use_rgb=True, image_weight = 0.5)
        outp_arr = np.asarray(outp)
        #side_by_side = np.concatenate((np.rot90(vol[i,:,:,:]), np.rot90(outp_arr)), axis=1)
        side_by_side = np.concatenate((np.rot90(vol[i,:,:,:]), np.rot90(outp_arr)), axis=1)
        
        side_by_side_frames.append(side_by_side)
    side_by_side_frames = np.array(side_by_side_frames)

    # output to output_file_path
    imageio.mimwrite(output_file_path, side_by_side_frames, format='GIF', fps=5)

# Get user input
global args
args = load_config()
set_defaults_config()

transform_fn = None

# load all models in
model_paths = glob.glob(args["model_path_pattern"])
models_list = []
for x in sorted(model_paths):
    model = model_prediction(x)
    models_list.append(model.cuda().eval())

# load dataset
dset = get_datasets_3d(dataset_type=args["dataset_type"], 
                        norm=args["normalize"], transform=transform_fn, 
                        data_path_img = args["data_path_img"], data_path = args["data_path_datafiles"],
                        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"], 
                        normalize_mean=args.get("normalize_mean", None), normalize_std=args.get("normalize_std", None),
                        ct_mode='CT', ct_bone_seg_dilation_erosion_pixels=args.get("ct_bone_seg_dilation_erosion_pixels", None), 
                        ct_bone_seg_dilation_alone_pixels=args.get("ct_bone_seg_dilation_alone_pixels", None))
bin_thresh = args['binarization_threshold']

model_probabilities, merged, merged_before_drop_mipview = load_merged_dfs(bin_thresh)
dfo_prep = pd.read_csv(args['scans_for_inspection_csv'])

# first, plot average interpretability over all the models
layer_idx = args["layer_idx"]
with warnings.catch_warnings():
    # for a single seed, do all interp's
    save_dir = f'{args["output"]}/{args["dataset_type"]}/interpretability/'
    
    for i, row in dfo_prep.iterrows():
        accession_id = row['accession']
        label_number = row['label']
        lab_type = row['type']
        true_label = row['true_label']
        
        #for method in ['GuidedGradCam', 'GuidedBackprop', 'pytorch-gradcam']:
        for method in ['medcam']:
        #for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop', 'pytorch-gradcam']:
        #for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop']:
        #for method in ['pytorch-gradcam', 'GuidedGradCam']:
        #for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop']:
        #for method in ['GuidedBackprop']:

            new_path = os.path.join(save_dir, lab_type, str(accession_id), f'{method}_explain{label_number}.gif')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            print(f"Processing {accession_id}, label={true_label}, explaining {label_number}, type {lab_type}, method {method}")
            inp_arr, ct_img, attr, targets = calculate_interpretability_for_study_3D(
                merged_before_drop_mipview,
                accession_id = accession_id, 
                label_number = label_number, 
                type=method, layer_to_iterate = layer_idx, 
                device='cuda',
                model_idx = range(10))
            visualize_3d_vol_and_attr(inp_arr, ct_img, attr, new_path)


