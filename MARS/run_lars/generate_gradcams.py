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

import utils
from dataset import get_datasets, dataset_center
from utils import model_prediction, my_model, model_prediction_dual_mipview, my_model_feat_fusion

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
        'combine_mipviews': None,
        'layer_idx': 4,
        'binarization_threshold': 0.5,
        'force_label_explain': None
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
    df = model_probabilities[model_probabilities['accession'].isin(split[args["dataset_type"]])]
    binarization_threshold = binarization_threshold # for now

    uses_mipview = 'mipview' in model_probabilities.columns
    print('Uses mipview: ', uses_mipview)
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

default_cmap = LinearSegmentedColormap.from_list('custom green',
                                                 [(0, '#ffffff'),
                                                  (1, "#08ff4a")], N=256)

cmap_2 = LinearSegmentedColormap.from_list('',
                                                 [(0, "#ff0000"),
                                                  (0.5, "#FFFFFF"),
                                                  (1, '#0000ff')], N=256)
def dataset_tensor_to_rgb_img(tensor):
    # tensor should have shape (C, 310, 310)
    nparray = torch.flatten(tensor, start_dim=0, end_dim=1).numpy() # (C*310,310)
    nparray = np.stack((nparray,)*3, axis=2) # (C*310,310,3) 
    nparray = (nparray - nparray.min()) / (nparray.max() - nparray.min())
    nparray = nparray * 255
    nparray = nparray.astype(np.uint8)
    return nparray

def calculate_interpretability_for_study(merged_df, accession_id, mipview, label_number, type='IntegratedGradients', layer_to_iterate = None, device='cuda', model_idx = None):
    pred_label_idx = label_number
    rr = dset.df[(dset.df['accession'] == accession_id) & (mipview is None or dset.df['mipview'] == mipview)]
    idx = rr.index[0]
    assert len(rr) == 1, "accession_id not found or not unique"
    rr = dset[idx][0] # get the tensor
    rgb_img = dataset_tensor_to_rgb_img(rr) # convert to rgb image
    input_shape_per_batch = rr.shape

    print(merged_df.shape)
    print(accession_id)
    print(mipview)
    print(merged_df[(merged_df['accession'] == accession_id) & (mipview is None or merged_df['mipview'] == mipview)])

    predprob = merged_df[(merged_df['accession'] == accession_id) & (mipview is None or merged_df['mipview'] == mipview)]['probs'].values[0]
    if model_idx is not None:
        predprob = model_probabilities[(model_probabilities['accession'] == accession_id) & (model_probabilities['model_number'].isin(model_idx)) & (mipview is None or model_probabilities['mipview'] == mipview)]['probs']
        predprob = predprob.mean()
        #assert len(predprob) == 1, "Prediction probability not found or not unique"
    inp_tensor = rr.unsqueeze(0).to(device)
    attribution_list = []
    inp_tensor.requires_grad=True
    for model_this_idx in model_idx:
        model_to_use = models_list[model_this_idx]
        model_to_use.cuda().eval()
        target_layer = model_to_use.features[layer_to_iterate][2].conv2

        if type == 'pytorch-gradcam':
            assert mipview is not None # dual-mipview cannot be used with gradcam because it averaging gradients in the channels is meaningless
            targets = [ClassifierOutputTarget(label_number)]
            
            with GradCAM(model=model_to_use, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=inp_tensor, targets=targets)

            attributions = np.reshape(grayscale_cam, (input_shape_per_batch[0] * input_shape_per_batch[1], input_shape_per_batch[2])) # remove channel dimension by flattening, for visualization against rgb_img
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
                                        sliding_window_shapes=(1, 8, 8), 
                                        strides=(1, 8, 8), 
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

    return (np.uint8(255 - rgb_img), attributions, predprob)

def visualize_interpretability(rgb_img, attributions, predprob, fig, axs, type='IntegratedGradients', alpha=0.7, i1=None, i2=None):
    cmap = None
    if type in ['IntegratedGradients', 'LayerGradCam', 'GuidedGradCam']:
        cmap = default_cmap
        sign = 'positive'
    elif type=='GuidedBackprop':
        cmap = cmap_2
        sign='all'
    else:
        cmap = None
        sign='all'
    
    method_viz = 'heat_map' if type=='GuidedBackprop' else 'blended_heat_map'
    if i1 is None:
        if type=='pytorch-gradcam':
            rgb_img_norm = rgb_img / 255.0
            visualization = show_cam_on_image(rgb_img_norm, attributions.squeeze(), use_rgb=True, image_weight = 0.5)
            axs.imshow(visualization)   
            axs.axis('off')
            return
        
        bd = viz.visualize_image_attr(
            np.transpose(attributions, (1,2,0)),
            rgb_img,
            method=method_viz,
            cmap=cmap,
            show_colorbar=False,
            sign=sign,
            title='',
            alpha_overlay = alpha,
            plt_fig_axis=(fig, axs), use_pyplot=False)
    else:
        A1 = attributions[:, :attributions.shape[1]//2, :]
        A2 = attributions[:, attributions.shape[1]//2 : -1, :]
        bd = viz.visualize_image_attr(
            np.transpose(A1, (1,2,0)),
            rgb_img[:rgb_img.shape[0]//2, :, :],
            method=method_viz,
            cmap=cmap,
            show_colorbar=False,
            sign=sign,
            title='',
            alpha_overlay = alpha,
            plt_fig_axis=(fig, axs[i1[0], i1[1]]), use_pyplot=False)
        bd = viz.visualize_image_attr(
            np.transpose(A2, (1,2,0)),
            rgb_img[rgb_img.shape[0]//2 : -1, :, :],
            method=method_viz,
            cmap=cmap,
            show_colorbar=False,
            sign=sign,
            title='',
            alpha_overlay = alpha,
            plt_fig_axis=(fig, axs[i2[0], i2[1]]), use_pyplot=False)
        

    
    assert rgb_img.min() >= 0 and rgb_img.max() <= 255 

# Get user input
global args
args = load_config()
set_defaults_config()

transform_fn = None

# load all models in
model_paths = glob.glob(args["model_path_pattern"])
models_list = []
for x in sorted(model_paths):
    if args['combine_mipviews'] is not None:
        model = model_prediction_dual_mipview(x)
    else:
        model = model_prediction(x, arch='resnet34')
    models_list.append(model.cuda().eval())

# load dataset
dset = get_datasets(dataset_type=args["dataset_type"], 
                        norm=args["normalize"], transform=transform_fn, 
                        data_path_img = args["data_path_img"], data_path = args["data_path_datafiles"],
                        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"], 
                        normalize_mean=args.get("normalize_mean", None), normalize_std=args.get("normalize_std", None),
                        combine_mipviews=args.get("combine_mipviews", None))
bin_thresh = args['binarization_threshold']

uses_mipview = args['combine_mipviews'] is None
model_probabilities, merged, merged_before_drop_mipview = load_merged_dfs(bin_thresh)
dfo_prep = pd.read_csv(args['scans_for_inspection_csv'])

force_label_explain = args['force_label_explain']

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

        if force_label_explain is not None:
            label_number = force_label_explain
            print(f'Forcing explanation of label {label_number} for accession {accession_id}')

        for method in ['IntegratedGradients', 'GuidedGradCam', 'Occlusion', 'GuidedBackprop', 'pytorch-gradcam']:
        #for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop', 'pytorch-gradcam']:
        #for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop']:
        #for method in ['pytorch-gradcam', 'GuidedGradCam']:
        #for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop']:
        #for method in ['GuidedBackprop']:
            if not uses_mipview and method == 'pytorch-gradcam': continue # not possible

            new_path = os.path.join(save_dir, lab_type, str(accession_id), f'{method}_explain{label_number}.png')
            if os.path.exists(new_path):
                print(f'Skipping {new_path}')
                continue

            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            fig, axs = plt.subplots(2,2, figsize = (10,10))

            # show original image
            
            print(f"Processing {accession_id}, label={true_label}, explaining {label_number}, type {lab_type}, method {method}")
            if uses_mipview:
                cor_img, cor_attribution, cor_prob = calculate_interpretability_for_study(merged_before_drop_mipview, accession_id, 'cor', label_number, type=method, layer_to_iterate=layer_idx, device='cuda',
                    model_idx = range(10))
                sag_img, sag_attribution, sag_prob  = calculate_interpretability_for_study(merged_before_drop_mipview, accession_id, 'sag', label_number, type=method, layer_to_iterate=layer_idx, device='cuda',
                    model_idx = range(10))
                visualize_interpretability(cor_img, cor_attribution, cor_prob, fig, axs[0,1], type=method, alpha=0.7)
                visualize_interpretability(sag_img, sag_attribution, sag_prob, fig, axs[1,1], type=method, alpha=0.7)
            else:
                cor_img, cor_attribution, cor_prob = calculate_interpretability_for_study(merged_before_drop_mipview, accession_id, None, label_number, type=method, layer_to_iterate=layer_idx, device='cuda',
                    model_idx = range(10))
                visualize_interpretability(cor_img, cor_attribution, cor_prob, fig, axs, type=method, alpha=0.7, i1=(0,1), i2=(1,1))
                sag_img = cor_img[cor_img.shape[0]//2 : -1,:,:]
                cor_img = cor_img[:cor_img.shape[0]//2,:,:]
                sag_prob = cor_prob
                
            axs[0,0].imshow(cor_img)
            axs[1,0].imshow(sag_img)

            fig.suptitle(f'{method} - accession {accession_id}, label={true_label}, explaining {label_number}, corprob= {cor_prob:.2f}, sagprob= {sag_prob:.2f}')

            #fig.savefig(new_path, dpi=600)

            for ii in range(2):
                for jj in range(2):
                    axs[ii, jj].axis('off')
            fig.savefig(new_path, dpi=600)
            plt.close()

# next, show per-model interp plots
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # compare interps across seeds
    layer_idx=args["layer_idx"]
    save_dir = f'{args["output"]}/{args["dataset_type"]}/interpretability_compare_seeds/'
    for i, row in dfo_prep.iterrows():
        accession_id = row['accession']
        label_number = row['label']
        lab_type = row['type']
        true_label = row['true_label']

        for method in ['IntegratedGradients', 'Occlusion', 'GuidedBackprop', 'GuidedGradCam', 'pytorch-gradcam']:
            if not uses_mipview and method == 'pytorch-gradcam': continue # not possible
        #for method in ['GuidedBackprop']:
            new_path = os.path.join(save_dir, lab_type, str(accession_id), f'{method}_explain{label_number}.png')
            if os.path.exists(new_path):
                print(f'Skipping {new_path}')
                continue
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            fig, axs = plt.subplots(2,11, figsize = (25,5))
            print(f"Processing {accession_id}, label={true_label}, explaining {label_number}, type {lab_type}, method {method}")
            sag_probs = []
            cor_probs = []
            
            for seed_idx, model_to_use in enumerate(models_list):
                print(f'Seed {seed_idx}')

                if uses_mipview:
                    cor_img, cor_attribution, cor_prob = calculate_interpretability_for_study(merged_before_drop_mipview, accession_id, 'cor', label_number, type=method, layer_to_iterate=layer_idx, device='cuda',
                        model_idx = [seed_idx])
                    sag_img, sag_attribution, sag_prob = calculate_interpretability_for_study(merged_before_drop_mipview, accession_id, 'sag', label_number, type=method, layer_to_iterate=layer_idx, device='cuda',
                        model_idx = [seed_idx])
                    visualize_interpretability(cor_img, cor_attribution, cor_prob, fig, axs[0,seed_idx+1], type=method, alpha=0.7)
                    visualize_interpretability(sag_img, sag_attribution, sag_prob, fig, axs[1,seed_idx+1], type=method, alpha=0.7)
                    
                    cor_probs.append(cor_prob)
                    sag_probs.append(sag_prob)
                    
                else:
                    cor_img, cor_attribution, cor_prob = calculate_interpretability_for_study(merged_before_drop_mipview, accession_id, None, label_number, type=method, layer_to_iterate=layer_idx, device='cuda',
                        model_idx = [seed_idx])
                    
                    visualize_interpretability(cor_img, cor_attribution, cor_prob, fig, axs, type=method, alpha=0.7, i1=(0,seed_idx+1), i2=(1,seed_idx+1) )
                    sag_img = cor_img[cor_img.shape[0]//2 : -1,:,:]
                    cor_img = cor_img[:cor_img.shape[0]//2,:,:]
                    cor_probs.append(cor_prob)
                    sag_probs.append(cor_prob)
                
                if seed_idx == 0:
                    axs[0,0].imshow(cor_img)
                    axs[1,0].imshow(sag_img)
            
            for ii in range(2):
                for jj in range(11):
                    axs[ii, jj].axis('off')

            # round all floats in cor_probs to 2 places
            cor_probs = [float(round(prob, 2)) for prob in cor_probs]
            sag_probs = [float(round(prob, 2)) for prob in sag_probs]
            if uses_mipview:
                fig.suptitle(f'{method} - accession {accession_id}, label={true_label}, explaining {label_number}, corprobs (mean {np.mean(cor_probs):.2f})= {cor_probs}, sagprobs (mean {np.mean(sag_probs):.2f})= {sag_probs}')
            else:
                probs = cor_probs
                fig.suptitle(f'{method} - accession {accession_id}, label={true_label}, explaining {label_number}, probs (mean {np.mean(probs):.2f})= {probs}')
            fig.tight_layout()
            fig.savefig(new_path, dpi=600)
            

