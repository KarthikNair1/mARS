import os
from PIL import Image
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
from matplotlib import pyplot as plt

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
        'layer_idx': 4,
        'binarization_threshold': 0.5,
        'prob_mode': 'mean',
        'target_label_name': 'PET_jco'
    }

    for key, value in default_dict.items():
        if key not in args:
            args[key] = value


global args
args = load_config()
set_defaults_config()

# load aggregate csv
preds = pd.read_csv(args['probabilities_aggregate_csv'])

# use probs_mean
col_list = ['filepath', 'filename', f'probs_{args["prob_mode"]}', args['target_label_name'], 'model_number']
preds = preds[col_list]
preds = preds.rename({'probs_' + args['prob_mode']: 'probs', args['target_label_name']: 'label'}, axis=1)
preds = preds.sort_values(by=['filepath', 'model_number']).reset_index(drop=True)

pred_collapse_model_number = preds.groupby(['filepath']).agg({'probs': 'mean', 'label': 'first', 'filename': 'first'}).reset_index()

binarization_threshold = args['binarization_threshold']
pred_collapse_model_number['pred_label'] = (pred_collapse_model_number['probs'] > binarization_threshold).astype(int)

# get error cases
false_negatives = pred_collapse_model_number[(pred_collapse_model_number['label'] == 1) & (pred_collapse_model_number['pred_label'] == 0)]
false_positives = pred_collapse_model_number[(pred_collapse_model_number['label'] == 0) & (pred_collapse_model_number['pred_label'] == 1)]

# create output directory
output_dir = os.path.join(args['output'], 'error_cases')
os.makedirs(output_dir, exist_ok=True)

# save false negatives
output_dir_fn = os.path.join(output_dir, 'false_negatives')
os.makedirs(output_dir_fn, exist_ok=True)

print(false_negatives)
print(len(preds))
print(f"Total number of cases: {len(pred_collapse_model_number)}")
print(f"Number of false negatives: {len(false_negatives)}")
print(f"Number of false positives: {len(false_positives)}")


for idx, row in false_negatives.iterrows():
    qc_rt = '/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_thumbnails_png_curated'
    qc_png_path = qc_rt + '/' + row['filename'].replace('.h5', '.png')
    assert os.path.exists(qc_png_path), f"QC PNG not found: {qc_png_path}"

    # loasd qc_png_path as numpy array
    qc_png = np.array(Image.open(qc_png_path))
    all_probs = preds[preds['filepath'] == row['filepath']]['probs'].values
    all_probs = [float(round(x, 2)) for x in all_probs]

    fig, axs = plt.subplots()
    axs.imshow(qc_png)
    axs.axis('off')
    axs.set_title(f'False Negative, True={row["label"]}\n Predicted Probabilities (mean={np.mean(all_probs):.2f}): {all_probs}', fontsize=7)
    fig.tight_layout()

    save_path = os.path.join(output_dir_fn, row['filename'].replace('.h5', '_err.png'))
    fig.savefig(save_path)
    plt.close(fig)

# save false positives
output_dir_fp = os.path.join(output_dir, 'false_positives')
os.makedirs(output_dir_fp, exist_ok=True)
for idx, row in false_positives.iterrows():
    qc_rt = '/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_thumbnails_png_curated'
    qc_png_path = qc_rt + '/' + row['filename'].replace('.h5', '.png')
    assert os.path.exists(qc_png_path), f"QC PNG not found: {qc_png_path}"

    # loasd qc_png_path as numpy array
    qc_png = np.array(Image.open(qc_png_path))
    all_probs = preds[preds['filepath'] == row['filepath']]['probs'].values
    assert preds[preds['filepath'] == row['filepath']]['model_number'].values.tolist() == list(range(10)), "Model numbers do not match expected range"
    all_probs = [float(round(x, 2)) for x in all_probs]

    fig, axs = plt.subplots()
    axs.imshow(qc_png)
    axs.axis('off')
    axs.set_title(f'False Positive, True={row["label"]}\n Predicted Probabilities (mean={np.mean(all_probs):.2f}): {all_probs}', fontsize=7)
    fig.tight_layout()

    save_path = os.path.join(output_dir_fp, row['filename'].replace('.h5', '_err.png'))
    fig.savefig(save_path)
    plt.close(fig)