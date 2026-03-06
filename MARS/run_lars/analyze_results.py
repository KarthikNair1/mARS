import pandas as pd
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix,balanced_accuracy_score
import pdb

# Files
target = pd.read_excel('/gpfs/data/mayerhoeferlab/projects/myeloma/curated_status/Myeloma_master_ref_15_08_2025.xlsx')
target = target.rename(columns={'ACC':'accession','PET-jco':'PET_jco','PET-imwg':'PET_imwg'})
df1 = pd.read_csv('predictions/ensemble_aucs.csv')
df2 = pd.read_csv('predictions_pf/ensemble_aucs.csv')

df1 = df1.merge(target,on='accession',how='left')
df2 = df2.merge(target,on='accession',how='left')

df1 = df1[~df1.PET_jco.isna()]
df2 = df2[~df2.PET_jco.isna()]
df1 = df1[(df1.PET_jco.isin([0,1])) | (df1.PET_imwg.isin([0,1]))]
df2 = df2[(df2.PET_jco.isin([0,1])) | (df2.PET_imwg.isin([0,1]))]

# target_col = 'PET_imwg'
target_col = 'PET_jco'

### Balanced accuracy
accbal_larsavg = balanced_accuracy_score(df1[target_col], df1['pred_larsavg'])
accbal_larsmax = balanced_accuracy_score(df1[target_col], df1['pred_larsmax'])
accbal_larsavg_pf = balanced_accuracy_score(df2[target_col], df2['pred_larsavg'])
accbal_larsmax_pf = balanced_accuracy_score(df2[target_col], df2['pred_larsmax'])

### Confusion matrix w/o PF
cm_larsavg = confusion_matrix(df1['pred_larsavg'], df1[target_col])#,normalize='true')
cm_larsmax = confusion_matrix(df1['pred_larsmax'], df1[target_col])#,normalize='true')
### Confusion matrix w/o PF
cmpf_larsavg = confusion_matrix(df2['pred_larsavg'], df2[target_col])#,normalize='true')
cmpf_larsmax =confusion_matrix(df2['pred_larsmax'], df2[target_col])#,normalize='true')

print('\n********************\nW/O POSTFILTERING:\n********************\n')
# print('LARS-avg:\n{}\n\n'.format(cm_larsavg))
# print('LARS-max:\n{}'.format(cm_larsmax))
# print('\nAccuracy: LARS-avg: {:.3f},  LARS-max: {:.3f}'.format(accbal_larsavg,accbal_larsmax))
# tn, fp, fn, tp = cm_larsavg.ravel()
# print('LARS-avg:\t ACC={:.3f},  TP={:.3f}, TN={:.3f}, FP={:.3f}, FN={:.3f}'.format(accbal_larsavg,tp,tn,fp,fn))
# tn, fp, fn, tp = cm_larsmax.ravel()
# print('LARS-max:\t ACC={:.3f},  TP={:.3f}, TN={:.3f}, FP={:.3f}, FN={:.3f}'.format(accbal_larsmax,tp,tn,fp,fn))
tn, fp, fn, tp = cm_larsavg.ravel()
sens, spec = (tp/(tp+fn)), (tn/(tn+fp))
print('LARS-avg:\t ACC={:.3f}, SENS={:.3f}, SPEC={:.3f},   TP={:.0f}, TN={:.0f}, FP={:.0f}, FN={:.0f}'.format(accbal_larsavg,sens,spec,tp,tn,fp,fn))
tn, fp, fn, tp = cm_larsmax.ravel()
sens, spec = (tp/(tp+fn)), (tn/(tn+fp))
print('LARS-max:\t ACC={:.3f}, SENS={:.3f}, SPEC={:.3f},   TP={:.0f}, TN={:.0f}, FP={:.0f}, FN={:.0f}'.format(accbal_larsmax,sens,spec,tp,tn,fp,fn))


print('\n********************\nW/ POSTFILTERING:\n********************\n')
# print('LARS-avg:\n{}\n\n'.format(cmpf_larsavg))
# print('LARS-max:\n{}\n\n'.format(cmpf_larsmax))
# print('\nAccuracy: LARS-avg: {:.3f},  LARS-max: {:.3f}'.format(accbal_larsavg_pf,accbal_larsmax_pf))
# tn, fp, fn, tp = cmpf_larsavg.ravel()
# print('LARS-avg:\t ACC={:.3f},  TP={:.3f}, TN={:.3f}, FP={:.3f}, FN={:.3f}'.format(accbal_larsavg_pf,tp,tn,fp,fn))
# tn, fp, fn, tp = cmpf_larsmax.ravel()
# print('LARS-max:\t ACC={:.3f},  TP={:.3f}, TN={:.3f}, FP={:.3f}, FN={:.3f}'.format(accbal_larsmax_pf,tp,tn,fp,fn))
tn, fp, fn, tp = cmpf_larsavg.ravel()
sens, spec = (tp/(tp+fn)), (tn/(tn+fp))
print('LARS-avg:\t ACC={:.3f}, SENS={:.3f}, SPEC={:.3f},   TP={:.0f}, TN={:.0f}, FP={:.0f}, FN={:.0f}'.format(accbal_larsavg_pf,sens,spec,tp,tn,fp,fn))
tn, fp, fn, tp = cmpf_larsmax.ravel()
sens, spec = (tp/(tp+fn)), (tn/(tn+fp))
print('LARS-max:\t ACC={:.3f}, SENS={:.3f}, SPEC={:.3f},   TP={:.0f}, TN={:.0f}, FP={:.0f}, FN={:.0f}'.format(accbal_larsmax_pf,sens,spec,tp,tn,fp,fn))