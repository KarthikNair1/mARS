import pandas as pd
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pdb
import argparse

parser = argparse.ArgumentParser(description='PET myeloma prediction with LARS')
parser.add_argument('--output', type=str, default='.', help='name of output folder')
args = parser.parse_args() 
print(args)

# Loop over files    
df = pd.read_csv(os.path.join(args.output,'pred_mipview_aggregate.csv'))

# Ensemble
filter_col = [col for col in df if col.startswith('F')]
ensemble = df.groupby(['filename'])[['probs_mean','probs_max']+filter_col].agg(['mean']).reset_index(drop=False)
ensemble.columns = ["_".join(x) for x in ensemble.columns.ravel()]
ensemble = ensemble.rename(columns={'filename_':'filename'})
print(len(df),len(ensemble))
# Calculate auc
# fpr, tpr, thresholds = roc_curve(ensemble.target.values, ensemble.probs_mean_min.values,pos_label=1)
# auc_min = auc(fpr,tpr)
# fpr, tpr, thresholds = roc_curve(ensemble.target.values, ensemble.probs_mean_max.values,pos_label=1)
# auc_max = auc(fpr,tpr)
# fpr, tpr, thresholds = roc_curve(ensemble.target.values, ensemble.probs_mean_mean.values,pos_label=1)
# auc_mean = auc(fpr,tpr)
# ensemble['auc_mean'] = auc_mean
# ensemble['auc_min'] = auc_min
# ensemble['auc_max'] = auc_max
# Accuracy
# bestidx = np.argmax(tpr - fpr)
# bestth = thresholds[bestidx]
bestth = 0.359923 # From LARS
ensemble['threshold'] = bestth
ensemble['pred_larsavg'] = (ensemble['probs_mean_mean']>=bestth).astype(int)
ensemble['pred_larsmax'] = (ensemble['probs_max_mean']>=bestth).astype(int)
# ensemble['correct_larsavg'] = (ensemble['pred_larsavg']==ensemble.target).astype(int)
# ensemble['correct_larsmax'] = (ensemble['pred_larsmax']==ensemble.target).astype(int)
# acc_mean = ensemble['correct_lars-avg'].mean()
# acc_mean_bal = ensemble.groupby('target')['correct_lars-avg'].agg('mean').mean()
# ensemble['acc_mean'] = acc_mean
# ensemble['acc_mean_bal'] = acc_mean_bal
# # Spec and sens
# tp = ((ensemble.target==1)&(ensemble.correct_mean==1)).sum()
# tn = ((ensemble.target==0)&(ensemble.correct_mean==1)).sum()
# fn = ((ensemble.target==1)&(ensemble.correct_mean==0)).sum()
# fp = ((ensemble.target==0)&(ensemble.correct_mean==0)).sum()
# sens = tp/(tp+fn)
# spec = tn/(tn+fp)
# ensemble['sens'] = sens
# ensemble['spec'] = spec
# print('{} set ensemble (top {:02d})  auc={:.4f}  acc={:.4f}  acc_bal={:.4f}  sens={:.4f}  spec={:.4f}  Youden={:.4f}'.format(os.getcwd().split('/')[-1].split('_')[0],n,auc_mean,acc_mean,acc_mean_bal,sens,spec,bestth))

# Info
ensemble = ensemble.merge(df[['filepath', 'filename', 'filename_header', 'fullfile', 'patient_age',
       'patient_sex', 'patient_weight', 'accession', 'study_uid', 'series_uid',
       'scan_time', 'modality', 'matrix_size_1', 'matrix_size_2',
       'matrix_size_3', 'pixel_size_1', 'pixel_size_2', 'pixel_size_3',
       'pixel_size_3_str', 'slice_thickness', 'description_series',
       'description_study', 'admitting_diagnoses', 'protocol_name',
       'body_part', 'manufacturer', 'manufacturer_model', 'software',
       'reconstruction', 'corrections', 'pixel_unit', 'factor_bqml2suv',
       'frame_duration', 'tracer', 'tracer_injection_time', 'tracer_dose',
       'tracer_halflife', 'contrast_agent', 'patient_position',
       'imaging_plane', 'image_orientation', 'patient_history', 'img_max',
       'img_min', 'img_mean', 'img_std', 'img_sum', 'img_count_nonzeros',
       'img_mean_nonzeros', 'img_std_nonzeros', 'img_sum_nonzeros',
       'convolution_kernel', 'image_filter', 'xray_tube_current', 'kvp',
       'image_comments', 'smoking_status', 'contrast_route']],on='filename',how='left')
ensemble.to_csv(os.path.join(args.output,'ensemble_aucs.csv'),index=False)

# Group to one file    
# SEED = pd.DataFrame()
# files = glob.glob(os.path.join(args.output,'pred_seed*.csv'))
# SEED = pd.DataFrame()
# for file in files:
#     tmp = pd.read_csv(file)
#     tmp['seed'] = int(file.split('seed')[1].split('.')[0])
#     # AUC
#     fpr, tpr, thresholds = roc_curve(tmp.target.values, tmp.probs.values,pos_label=1)
#     ac = auc(fpr,tpr)
#     tmp['auc'] = ac
#     # ACC
#     bestidx = np.argmax(tpr - fpr)
#     bestth = thresholds[bestidx]
#     tmp['threshold'] = bestth
#     tmp['pred'] = (tmp['probs']>bestth).astype(int)
#     tmp['correct'] = (tmp.pred==tmp.target).astype(int)
#     acc = tmp.correct.mean()
#     acc_bal = tmp.groupby('target').correct.agg('mean').mean()
#     tmp['acc'] = acc
#     tmp['acc_bal'] = acc_bal
#     tp = ((tmp.target==1)&(tmp.correct==1)).sum()
#     tn = ((tmp.target==0)&(tmp.correct==1)).sum()
#     fn = ((tmp.target==1)&(tmp.correct==0)).sum()
#     fp = ((tmp.target==0)&(tmp.correct==0)).sum()
#     sens = tp/(tp+fn)
#     spec = tn/(tn+fp)
#     tmp['sens'] = sens
#     tmp['spec'] = spec
    
#     SEED = SEED.append(tmp,sort=False,ignore_index=True)

# # Info
# SEED = SEED.sort_values(by=['filename','seed'])
# SEED.to_csv(os.path.join(args.output,'seed_aucs.csv'),index=False)