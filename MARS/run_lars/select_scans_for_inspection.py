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
        'prob_mode': 'mean',
        'target_label_name': 'PET_jco',
        'binarization_threshold': 0.5
    }

    for key, value in default_dict.items():
        if key not in args:
            args[key] = value

def save_copy_of_config(filename):
    config_path = sys.argv[1]
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    dest_path = os.path.join(args["output"], filename)
    shutil.copyfile(config_path, dest_path)
    print(f"Configuration file copied to {dest_path}")

global args
args = load_config()
set_defaults_config()

# load overall probabilities
df = pd.read_csv(args["probabilities_aggregate_csv"])
df = df[['filepath', args["target_label_name"], f'probs_{args["prob_mode"]}']]
df = df.rename(columns={'filepath': 'accession', f'probs_{args["prob_mode"]}': 'probs', args["target_label_name"]: 'true_label'})
col_names = df.columns.tolist()
df = df.groupby(['accession', 'true_label']).agg({'probs': 'mean'}).reset_index()

df['label']  = (df['probs'] >= args["binarization_threshold"]).astype(int)

# populate "type" column
df['type'] = df.apply(lambda row: 'TP' if (row['label'] == 1 and row['true_label'] == 1) else ('TN' if (row['label'] == 0 and row['true_label'] == 0) else ('FP' if (row['label'] == 1 and row['true_label'] == 0) else 'FN')), axis=1)
print(df.value_counts('type'))

choose = {'TP': 15, 'FP': 1000000000, 'FN': 5, 'TN': 5}

dfo_list = []
for tag in ['TP', 'FP', 'FN', 'TN']:
    df_tag = df[df['type'] == tag]
    num_need = min(choose[tag], len(df_tag))

    df_tag = df_tag.sort_values(by = 'probs', ascending = (tag in ['FN', 'TN']))
    df_tag_selected = df_tag.head(num_need)
    dfo_list.append(df_tag_selected)

dfo_prep = pd.concat(dfo_list, axis=0).reset_index(drop=True)
dfo_prep = dfo_prep[['accession', 'label', 'type', 'true_label']]
dfo_prep.to_csv(os.path.join(args["output"], 'selected_scans_for_inspection.csv'), index=False)