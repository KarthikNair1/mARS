import pandas as pd
import glob
import os
import argparse
import re

parser = argparse.ArgumentParser(description='PET myeloma prediction with LARS')

parser.add_argument('--pred_file_pattern', type=str, default='.', help='pattern to prediction csv files, e.g. pred_*.csv')
args = parser.parse_args() 
print(args)

# Loop over files    
files = glob.glob(args.pred_file_pattern)
assert len(files) >= 1, 'No files found for pattern: {}'.format(args.pred_file_pattern)
assert 'pred_' in files[0], 'Pattern should contain "pred", e.g. pred_*.csv'
output_folder = os.path.dirname(files[0])

df = pd.DataFrame()
for i,file in enumerate(files):
    tmp = pd.read_csv(file)
    basefilename = os.path.basename(file)
    if basefilename == 'pred_test_mipview_aggregate.csv' or basefilename == 'pred_val_mipview_aggregate.csv':
        continue
    match = re.search(r"\d+", basefilename)
    assert match, 'Could not extract model number from filename: {}'.format(basefilename)
    model_number = int(match.group())

    tmp['model_number'] = model_number
    df = pd.concat([df,tmp],sort=False,ignore_index=True)
print('Read {} files.'.format(len(files)))

#1/0
# Max and min
filter_col = [col for col in df if col.startswith('F')]
tmp = df[['filename','probs','model_number']+filter_col].groupby(['filename','model_number']).aggregate(['mean','max'])
tmp.columns = ["_".join(x) for x in tmp.columns.ravel()]
tmp = tmp.reset_index(drop=False)
cols_to_drop = filter_col + ['probs']
if 'mipview' in df.columns:
    cols_to_drop.append('mipview')
dfm = df.drop(cols_to_drop, axis=1).drop_duplicates().merge(tmp, on=['filename','model_number'],how='left').reset_index(drop=True)
# Write to file
if 'val' in os.path.basename(files[0]):
    dfm.to_csv(os.path.join(output_folder,'pred_val_mipview_aggregate.csv'),index=False)
elif 'test' in os.path.basename(files[0]):
    dfm.to_csv(os.path.join(output_folder,'pred_test_mipview_aggregate.csv'),index=False)
else:
    dfm.to_csv(os.path.join(output_folder,'pred_mipview_aggregate.csv'),index=False)

print('Size aggregate:',len(df),'-->',len(dfm),'(',dfm.filename.nunique(),')')