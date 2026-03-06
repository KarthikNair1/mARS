import numpy as np
import pandas as pd
import os
import glob
import scipy
import torch
import torch.nn as nn
import pdb
import random
import itertools
from sklearn.metrics import roc_auc_score
import torchvision
import torch.nn.functional as F
from scipy import interpolate
import h5py
from scipy.ndimage import convolve1d, gaussian_filter
import wandb
import matplotlib.pyplot as plt
import math

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution, GuidedGradCam
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz

def calculate_metrics_from_probs(y_probs, y_true):
    y_pred = (y_probs >= 0.5).astype(int)
    accuracy = (y_true == y_pred).mean()
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = float('nan')

    # calculate fpr and fnr
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    ber = (fpr + fnr) / 2

    results = {
        'auc': auc,
        'accuracy': accuracy,
        'ber': ber,
        'fpr': fpr,
        'fnr': fnr
    }
    return results

def log_metrics(wandb_run, metrics, epoch, train_step, prefix=''):
    tmp = {}
    # rename dict
    for key, value in metrics.items():
        tmp[f'{prefix}{key}'] = value
    tmp['epoch'] = epoch
    tmp['train_step'] = train_step
    wandb_run.log(tmp)
    

def log_train_example_images_3D(wandb_run, train_ds, n=25):
    # just log the images
    n = min(n, len(train_ds))
    images = []
    nr = int(n**0.5)
    nc = math.ceil(n**0.5)
    fig, axs = plt.subplots(nr, nc, figsize=(25, 25))
    for i in range(n):
        r = i // nc
        c = i % nc
        vol, _ = train_ds[i]

        # pick random channel if multiple channels
        img_list = []
        for ch_idx in range(vol.shape[0]):
            img = vol[ch_idx, :, :, :].numpy()
            img = img[img.shape[0]//2, :, :]  # middle slice, coronal
            img = np.rot90(img)
            img = (img - img.min()) / (img.max() - img.min())
            img_list.append(img)

        # create concat image
        img = np.concatenate(img_list, axis=1)  # concatenate along width
        axs[r,c].imshow(img, cmap='gray_r')
        axs[r, c].axis('off')
        
    wandb_run.log({"train_examples": wandb.Image(fig)})

def log_val_example_images_3D(wandb_run, val_ds, val_probs, val_ground_truth, model, n_table_ex=10, n=5):
    # log n example images from the validation set
    n_table_ex = max(n_table_ex, n)
    table_indices = list(range(min(n_table_ex, len(val_ds))))
    images = []
    labels = []
    for idx in table_indices:
        img, label = val_ds[idx]
        images.append(img.numpy())
        labels.append(label)
        assert label == val_ground_truth[idx], f"Label mismatch: {label} vs {val_ground_truth[idx]}"
    
    # convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # make table
    columns = ["id", "image", "truth", "pred", "prob", 'prediction_type']
    table = wandb.Table(columns=columns)

    for i in range(n_table_ex):
        id = val_ds.df.loc[table_indices[i], 'accession']
        mask = val_ds.df['accession'] == id

        truth = labels[i]
        prob = val_probs[table_indices[i]]
        pred = int(prob >= 0.5)
        
        vol = val_ds[table_indices[i]][0][0,:,:,:].numpy() # (H, W, D), this is getting the 0th channel only
        cor_img = np.max(vol, axis=0)  # MIP coronal
        sag_img = np.max(vol, axis=1)  # MIP sagittal

        img_dict = {'coronal': cor_img, 'sagittal': sag_img}

        for view, img in img_dict.items():
            img = np.rot90(img)
            img = (img - img.min()) / (img.max() - img.min())  # normalize to [0,1] for better visualization
            img = 1 - img  # to reverse the color scale
            img_dict[view] = img

        # classify type of error using pred and truth
        if pred == 1 and truth == 0:
            error_type = 'FP'
        elif pred == 0 and truth == 1:
            error_type = 'FN'
        elif pred == 1 and truth == 1:
            error_type = 'TP'
        else:
            error_type = 'TN'

        # concatenate images side by side
        combined_img = np.concatenate([img_dict['coronal'], img_dict['sagittal']], axis=0)  # axis=0 for horizontal

        table.add_data(id, wandb.Image(combined_img), 
                       int(truth), int(pred), float(prob), error_type)
    
    wandb_run.log({"val_examples": table})

def log_classification_plots(wandb_run, probs, ground_truth, prefix = ''):

    probs_reshaped = np.stack([1 - probs, probs], axis=1)
    # ROC curve
    wandb_run.log({f"{prefix}val_roc": wandb.plot.roc_curve(ground_truth, probs_reshaped)})

    # confusion matrix
    wandb_run.log({f"{prefix}val_conf_mat": wandb.plot.confusion_matrix(y_true=ground_truth, preds=(probs >= 0.5).astype(int), class_names=['JCO_0', 'JCO_1'])})

def log_probability_histogram(wandb_run, probs, ground_truth):

    fig, axs = plt.subplots()

    _, bins, _ = axs.hist(probs[ground_truth==0], bins=50, range=[0, 1], density=True)
    _ = axs.hist(probs[ground_truth==1], bins=bins, alpha=0.5, density=True)

    axs.set_xlabel('Predicted probability')
    axs.set_ylabel('Count')
    axs.legend()
    wandb_run.log({"val_prob_hist": wandb.Image(fig)})

def log_learning_rates(wandb_run, optimizer, epoch):
    names = [param_group['name'] for param_group in optimizer.param_groups]
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    lr_dict = {f'lr/{name}': lr for name, lr in zip(names, lrs)}
    lr_dict['epoch'] = epoch
    wandb_run.log(lr_dict)

def get_log_learning_rates_step(wandb_run, optimizer, train_step):
    names = [param_group['name'] for param_group in optimizer.param_groups]
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    lr_dict = {f'lr/{name}': lr for name, lr in zip(names, lrs)}
    lr_dict['train_step'] = train_step
    return lr_dict

def log_data_distribution(all_values, wandb_run, prefix=''):
    # log:
    # 1. histogram of values
    # 2. mean and std of values
    fig, axs = plt.subplots()
    
    axs.hist(all_values, bins=100, range=[min(all_values), max(all_values)], density=True)
    axs.set_xlabel('Value')
    axs.set_ylabel('Density')
    wandb_run.log({f"{prefix}data_value_hist": wandb.Image(fig)})

    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    wandb_run.log({f"{prefix}data_value_mean": mean_val,
                    f"{prefix}data_value_std": std_val})
    