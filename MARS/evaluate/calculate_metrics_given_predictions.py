import pandas as pd
from sklearn import metrics, calibration
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def calculate_optimal_threshold_youden_index(df, prob_cal, label_col = 'PET_jco'):
    fpr, tpr, thresholds = metrics.roc_curve(df[label_col], df[prob_cal])
    youden_index = tpr - fpr
    optimal_idx = youden_index.argmax()
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


parser = argparse.ArgumentParser(description='PET metric calculation')

#I/O PARAMS
parser.add_argument('--pred_csv', type=str, default='.', help='path to csv with predictions')
parser.add_argument('--results_save_dir', type=str, default='.', help='directory to save results')
parser.add_argument('--train_test_split', type=str, default=None, help='pickle file to train/val/test split')
parser.add_argument('--which_split', type=str, default=None, help='which split to use', choices=['train', 'val', 'test'])
parser.add_argument('--binarization_threshold', type=float, default = None)

args = parser.parse_args()   

df = pd.read_csv(args.pred_csv)
results_save_dir = args.results_save_dir

# only use accessions in the train split
if args.train_test_split is not None:
    split = pd.read_pickle(args.train_test_split)
    df = df[df['accession'].isin(split[args.which_split])]

prob_cols = [col for col in df.columns if 'probs' in col]

df = df[['accession'] + prob_cols]

# average prob_cols using model_number to group
grouped = df.groupby(['accession'])

# summarize grouped df by taking first for every column, except prob_means, where we should take mean
summary_df = grouped.agg({
    'probs_mean': 'mean',  # Calculate the mean for 'probs_mean',
    'probs_max': 'mean',  # Calculate the mean for 'probs_max',
    **{col: 'first' for col in grouped.obj.columns if col not in ['probs_mean', 'probs_max']}  # Take the first value for all other columns
}).reset_index(drop=True)

df = summary_df

target = pd.read_excel('/gpfs/data/mayerhoeferlab/projects/myeloma/curated_status/Myeloma_master_ref_15_08_2025.xlsx')
target = target.rename(columns={'ACC':'accession','PET-jco':'PET_jco','PET-imwg':'PET_imwg'})
target = target[['accession', 'PET_jco', 'PET_imwg']]

# merge to get labels
merged = pd.merge(df, target[['accession', 'PET_jco', 'PET_imwg']] , on='accession', how='left')
merged = merged[['accession'] + prob_cols + ['PET_jco', 'PET_imwg']]

# exclude cases where there was no label available
merged = merged.dropna(subset = ['PET_jco'], axis='index')

# also exclude cases where the label is 2 (just for now)
merged = merged[merged['PET_jco'] != 2]

# find the optimal threshold by optimizing Youden's index
if args.binarization_threshold is None:
    binarization_threshold_mean = calculate_optimal_threshold_youden_index(merged, 'probs_mean', label_col = 'PET_jco')
    binarization_threshold_max = calculate_optimal_threshold_youden_index(merged, 'probs_max', label_col = 'PET_jco')
else:
    binarization_threshold_mean = args.binarization_threshold
    binarization_threshold_max = args.binarization_threshold

binarization_thresholds = {'probs_mean': binarization_threshold_mean, 'probs_max': binarization_threshold_max}

# save print output to a text file
if not os.path.exists(results_save_dir):
    os.makedirs(results_save_dir)
import sys
original_stdout = sys.stdout
f = open(f'{results_save_dir}/calculated_metrics.txt', 'w')
sys.stdout = f

# create binary predictions based on threshold
for prob_col in prob_cols:

    binarization_threshold = binarization_thresholds[prob_col]
    split_list = prob_col.split('probs')
    suffix = split_list[1] if len(split_list) > 1 else ''

    print(f'For {prob_col}:')
    print(f'Using binarization threshold: {binarization_threshold}')

    merged[f'pred{suffix}'] = (merged[prob_col]>=binarization_threshold).astype(int)


    print('Predictions:')
    print(f'Number of positive predictions: {merged[f"pred{suffix}"].sum()}')
    print(f'Number of negative predictions: {(merged[f"pred{suffix}"]==0).sum()}')

    accuracy = metrics.accuracy_score(merged['PET_jco'], merged[f'pred{suffix}'])
    balanced_accuracy = metrics.balanced_accuracy_score(merged['PET_jco'], merged[f'pred{suffix}'])
    confusion = metrics.confusion_matrix(merged['PET_jco'], merged[f'pred{suffix}'])
    report = metrics.classification_report(merged['PET_jco'], merged[f'pred{suffix}'])

    auc = metrics.roc_auc_score(merged['PET_jco'], merged[prob_col])
    print('Metrics:')
    print(f'Confusion matrix:\n{confusion}')
    print(f'Classification report:\n{report}')
    print(f'AUC: {auc}')

    # additionally, calculate sensitivity @ 95% specificity, and specificity at 95% sensitivity
    fpr, tpr, thresholds = metrics.roc_curve(merged['PET_jco'], merged[prob_col])
    sensitivities = tpr
    specificities = 1 - fpr
    # sensitivity at 95% specificity
    at_95spec_idx = np.abs(specificities - 0.95).argmin()
    sens_at_95spec = sensitivities[at_95spec_idx]
    spec_at_95spec = specificities[at_95spec_idx]
    threshold_95spec = thresholds[at_95spec_idx]
    # specificity at 95% sensitivity
    at_95sens_idx = np.abs(sensitivities - 0.95).argmin()
    sens_at_95sens = sensitivities[at_95sens_idx]
    spec_at_95sens = specificities[at_95sens_idx]
    threshold_95sens = thresholds[at_95sens_idx]
    print(f'Sensitivity at 95% specificity: Sens - {sens_at_95spec}, Spec - {spec_at_95spec}, threshold: {threshold_95spec}')
    print(f'Specificity at 95% sensitivity: Sens - {sens_at_95sens}, Spec - {spec_at_95sens}, threshold: {threshold_95sens}')
        
    # create ROC curve
    fig, ax = plt.subplots()
    fpr, tpr, _ = metrics.roc_curve(merged['PET_jco'], merged[prob_col])
    ax.plot(fpr, tpr, label = f'auc={str(auc)}')
    ax.set_title(f'ROC curve for {suffix}')
    fig.legend()

    # save to results_save_dir
    fig.savefig(f'{results_save_dir}/roc_curve_train{suffix}.png', dpi=600)

    # create confusion matrix
    fig, ax = plt.subplots()

    # palette:
    # white to #8F1E63

    #metrics.ConfusionMatrixDisplay(confusion, display_labels = ['JCO_0', 'JCO_1']).plot(ax=ax)
    cmap = sns.cubehelix_palette(as_cmap=True)
    sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap, xticklabels=['JCO_0', 'JCO_1'], yticklabels=['JCO_0', 'JCO_1'], ax=ax)
    # set ax x title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Predictions for {suffix}')
    fig.savefig(f'{results_save_dir}/confusion_matrix_train{suffix}.png', dpi=600)


    # create probability calibration plot

    num_bins = 7
    fig, ax = plt.subplots(2, 1, figsize = (6, 12))
    prob_true, prob_pred = calibration.calibration_curve(merged['PET_jco'], merged[prob_col], n_bins=num_bins, strategy='quantile')
    ax[0].plot(prob_pred, prob_true, marker='o')
    ax[0].plot([0,1],[0,1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax[0].set_xlabel('Mean predicted probability in bin')
    ax[0].set_ylabel('Fraction of positives in bin')
    ax[0].set_title(f'Model Probability Calibration - {suffix}')
    ax[0].legend()
    # plot histogram of predicted probabilities
    ax[1].hist(merged[prob_col], bins=num_bins, alpha=0.7, color='#454bcf', label='Predicted probabilities')
    ax[1].set_xlabel('Predicted probability')
    ax[1].set_ylabel('Count')

    fig.savefig(f'{results_save_dir}/probability_calibration_train{suffix}.png', dpi=600)


sys.stdout = original_stdout
f.close()