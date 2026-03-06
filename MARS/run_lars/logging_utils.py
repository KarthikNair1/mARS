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
    

def log_train_example_images(wandb_run, train_ds, n=25):
    # just log the images
    n = min(n, len(train_ds))
    images = []
    nr = int(n**0.5)
    nc = math.ceil(n**0.5)
    fig, axs = plt.subplots(nr, nc, figsize=(25, 25))
    for i in range(n):
        r = i // nc
        c = i % nc
        img, _ = train_ds[i]
        
        if img.shape == (2, 310, 310):
            # combine the coronal and saggital views into one image
            img = torch.cat([img[0,:,:], img[1,:,:]], dim=1)  # concatenate along width
        images.append(img.numpy())
        img = (img - img.min()) / (img.max() - img.min())
        img = 1 - img  # to reverse the color scale
        axs[r, c].imshow(img.squeeze(), cmap='gray')
        axs[r, c].axis('off')
    wandb_run.log({"train_examples": wandb.Image(fig)})

def log_val_example_images(wandb_run, val_ds, val_probs, val_ground_truth, model, n_table_ex=10, n=5, combine_mipviews=False):
    if combine_mipviews:
        print('Logging val table for combined mipview not implemented yet.')
        return
    # log n example images from the validation set
    n_table_ex = max(n_table_ex, n)
    table_indices = list(range(min(n_table_ex, len(val_ds))))
    indices = list(range(min(n, len(val_ds))))
    images = []
    labels = []
    for idx in table_indices:
        img, label = val_ds[idx]
        if isinstance(img, tuple):  # intended for combine_mipview='feature_concatenation', where each tuple element represents cor or sag image
            img = torch.cat([img[0][0,:,:], img[1][0,:,:]], dim=1)  # concatenate along width
        if img.shape == (2, 310, 310):
            # combine the coronal and saggital views into one image
            img = torch.cat([img[0,:,:], img[1,:,:]], dim=1)  # concatenate along width
        images.append(img.numpy())
        labels.append(label)
        assert label == val_ground_truth[idx], f"Label mismatch: {label} vs {val_ground_truth[idx]}"
    
    # convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # make table
    columns = ["id", "image", "truth", "pred", "coronal_prob", "sagittal_prob", "mean_prob", 'prediction_type']
    table = wandb.Table(columns=columns)

    for i in range(n_table_ex):
        id = val_ds.df.loc[table_indices[i], 'accession']
        mask = val_ds.df['accession'] == id

        print(val_ds.df[mask])

        coronal_index = val_ds.df.index[mask & (val_ds.df['mipview'] == 'cor')].tolist()
        sagittal_index = val_ds.df.index[mask & (val_ds.df['mipview'] == 'sag')].tolist()
        assert len(coronal_index) == 1, f"Multiple or no coronal images found for {id}"
        assert len(sagittal_index) == 1, f"Multiple or no sagittal images found for {id}"
        coronal_index = coronal_index[0]
        sagittal_index = sagittal_index[0]

        truth = labels[i]
        coronal_prob = val_probs[coronal_index]
        sagittal_prob = val_probs[sagittal_index]
        mean_prob = (coronal_prob + sagittal_prob) / 2

        pred = int(mean_prob >= 0.5)

        img_dict = {'coronal': val_ds[coronal_index][0].numpy(), 'sagittal': val_ds[sagittal_index][0].numpy()}

        for view, img in img_dict.items():
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
        combined_img = np.concatenate([img_dict['coronal'], img_dict['sagittal']], axis=1)  # axis=1 for horizontal

        table.add_data(id, wandb.Image(combined_img), 
                       int(truth), int(pred), float(coronal_prob), float(sagittal_prob), float(mean_prob), error_type)
    
    wandb_run.log({"val_examples": table})

    # now log the gradcams

    for i in range(n):
        obj = GuidedGradCam(model, model.features[7][2].conv2)  # last conv layer in resnet34
        inp_tensor = val_ds[indices[i]][0].clone().detach()
        inp_tensor = inp_tensor.view(1, 1, 310, 310).to('cuda')
        this_pred = int(val_probs[indices[i]] >= 0.5)
        attributions = obj.attribute(inp_tensor, target=this_pred)
        cmap = None
        sign='positive'

        # rgb image
        img_show = images[indices[i]].squeeze()
        img_show = np.stack((img_show,)*3, axis=2)
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())  # normalize to [0,1] for better visualization
        img_show = (img_show * 255).astype(np.uint8)

        attributions = attributions.cpu().detach().squeeze().view((1, 310, 310)).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].imshow(255 - img_show)
        axs[0].axis('off')


        try:
            _ = viz.visualize_image_attr(
                np.transpose(attributions, (1,2,0)),
                255 - img_show,
                method='blended_heat_map',
                cmap=cmap,
                show_colorbar=True,
                sign=sign,
                title=f'{val_ds.df.loc[indices[i], "accession"]} - True: {labels[i]}, Pred: {this_pred}, Prob: {val_probs[indices[i]]:.3f}',
                alpha_overlay = 0.7,
                plt_fig_axis=(fig, axs[1]))
        except Exception as e:
            # handle exception where attributions is all zeros
            print(f"Error visualizing attributions for index {indices[i]} with sum {attributions.sum()}: {e}")
            
        wandb_run.log({f"val_gradcam_{i}": wandb.Image(fig)})


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