import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle

# currently removing any studies without a label
# also still including any low quality scans
# stratifying based on PET_jco


seed = 2025
save_path = '/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/'
data_file_path = os.path.join(save_path, 'data_scan_curated.csv')
label_file_path = '/gpfs/data/mayerhoeferlab/projects/myeloma/curated_status/Myeloma_master_ref_15_08_2025.xlsx'

print(f'Seed = {seed}')
df = pd.read_csv(data_file_path)

# get the label file 
target = pd.read_excel(label_file_path)
target = target.rename(columns={'ACC':'accession','PET-jco':'PET_jco','PET-imwg':'PET_imwg'})

# drop rows of target where PET_jco = 2
target = target[target['PET_jco'] != 2]

# figure out which patient accessions have more than 1 scan, and drop those patients
accession_counts = target['accession'].value_counts()
multiple_accessions = accession_counts[accession_counts > 1].index
print('Accessions with more than 1 study:')
print(multiple_accessions.tolist())

df = df[~df['accession'].isin(multiple_accessions)]
target = target[~target['accession'].isin(multiple_accessions)]
print(f'After removing patients with more than 1 study, we have {df.shape[0]} scans from {df["accession"].nunique()} patients')

merged_df = pd.merge(df, target[['accession', 'PET_jco', 'PET_imwg']] , on='accession', how='left')
merged_df = merged_df.dropna(subset = ['PET_jco'], axis='index')

print(f'After merging, we have {merged_df.shape[0]} scans from {merged_df["accession"].nunique()} patients')

train_val, test = train_test_split(merged_df, train_size = .6, test_size = .4, random_state=seed, stratify = merged_df['PET_jco'], shuffle=True)
train, val = train_test_split(train_val, train_size = 5.0/6, test_size = 1.0/6, random_state=seed, stratify = train_val['PET_jco'], shuffle=True)

print(f'Train: {train.shape[0]} scans from {train["accession"].nunique()} patients')
print(f'Val: {val.shape[0]} scans from {val["accession"].nunique()} patients')
print(f'Test: {test.shape[0]} scans from {test["accession"].nunique()} patients')

# print distribution of labels in each split
print('Train label distribution:')
print(train['PET_jco'].value_counts(normalize=True))
print('Val label distribution:')
print(val['PET_jco'].value_counts(normalize=True))
print('Test label distribution:')
print(test['PET_jco'].value_counts(normalize=True))

# ensure there is no overlap between any of the patients included in each split
print( set(train['accession']).intersection(set(val['accession'])))
print( set(train['accession']).intersection(set(test['accession'])))
print( set(val['accession']).intersection(set(test['accession'])))
assert len(set(train['accession']).intersection(set(val['accession']))) == 0
assert len(set(train['accession']).intersection(set(test['accession']))) == 0
assert len(set(val['accession']).intersection(set(test['accession']))) == 0

# save dataframes
train.to_csv(os.path.join(save_path, 'train_df.csv'), index=False)
val.to_csv(os.path.join(save_path, 'val_df.csv'), index=False)
test.to_csv(os.path.join(save_path, 'test_df.csv'), index=False)

# save list of accession numbers for each split
holder = {
    'train': train['accession'].values.tolist(),
    'val': val['accession'].values.tolist(),
    'test': test['accession'].values.tolist()
}

os.makedirs(os.path.join(save_path, 'splits'), exist_ok=True)
with open(os.path.join(save_path, 'splits', 'train_val_test_split.pkl'), 'wb') as f:
    pickle.dump(holder, f)
