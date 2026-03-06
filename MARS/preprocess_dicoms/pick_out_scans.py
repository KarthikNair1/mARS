import pandas as pd
import numpy as np
import os
import h5py
import glob
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import fnmatch
import pdb
import argparse
from utils import show_mips, show_grid, bbox_nd, read_image_h5, read_header_pkl, get_sample_from_df

plt.rcParams.update({'figure.max_open_warning': 0})
def get_diff_from_prev(prev, curr):
    return set(prev['filename'].values) - set(curr['filename'].values)

def check_image(tmp2):
    tmp2 = tmp2.rename(columns={'patient_position':'pp','pixel_size_1':'ps1','pixel_size_2':'ps2','pixel_size_3':'ps3','matrix_size_1':'ms1','matrix_size_2':'ms2','matrix_size_3':'ms3','axial_fov':'fov','img_mean_nonzeros':'avg','reconstruction':'recon','protocol_name':'prot','description_series':'descr'})
    print(tmp2[['filename','descr','prot','recon','avg','ps1','ps3','ms1','ms3','fov',]].round(3))
    print(tmp2.descr.values)

    tmp,img,h,ps = ['']*len(tmp2),['']*len(tmp2),['']*len(tmp2),['']*len(tmp2)
    for i in range(len(tmp2)):
        #pdb.set_trace()
        tmp[i],img[i],h[i],ps[i] = get_sample_from_df(tmp2.iloc[i])
        show_mips(img[i],pixelsize=ps[i],scale=4,savename='test_'+str(i))
    return tmp,img,h,ps

parser = argparse.ArgumentParser(description='Select relevant scans from all scans.')
parser.add_argument('--input_csv', type=str, default='data_scan.csv', help='Path to the input CSV file')
parser.add_argument('--pixel_size_csv', type=str, default='data_count_axial_pixel_size.csv', help='Path to the axial pixel size CSV file')
parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
parser.add_argument('--faulty_scan_csv', type=str, default = 'data_faulty_scan.csv', help='Path to the faulty scans CSV file')
parser.add_argument('--thumbnail_save_dir', type=str, default = '/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_thumbnails_png_curated', help='Path to the curated thumbnail save directory')
args = parser.parse_args()

### Read PET data
df = pd.read_csv(args.input_csv)
cnt = pd.read_csv(args.pixel_size_csv)
#1/0
# Merge
df = df.merge(cnt[['filename','n_ps','n_ps2','n_ps3','n_ps4']],on='filename',how='left')

#1/0
# Remove faulty scans (manually curated)
faulty = pd.read_csv(args.faulty_scan_csv)
prev = df
df = df[~df.filename.isin(faulty.filename)]
print('Removed manually:')
print(get_diff_from_prev(prev=prev, curr=df))
# Only PET
prev = df
df = df[df.modality=='PT']
print('Removed non-PET:')
print(get_diff_from_prev(prev=prev, curr=df))
# BQML --> SUV
prev = df
df = df[df.pixel_unit=='BQML']
print('Removed non-BQML:')
print(get_diff_from_prev(prev=prev, curr=df))
# Transverse
prev = df
df = df[df.imaging_plane=='transverse']
print('Removed non-transverse:')
print(get_diff_from_prev(prev=prev, curr=df))
# Original 
prev = df
df = df[~df.description_series.str.contains('reformat',case=False)]
print('Removed reformats:')
print(get_diff_from_prev(prev=prev, curr=df))

#1/0
# Not too few slices
prev = df
df = df[df.matrix_size_3>20]
print('Removed if number of slices <= 20:')
print(get_diff_from_prev(prev=prev, curr=df))
# Fix 0 mm slice thickness
df.loc[df.pixel_size_3==0,'pixel_size_3'] = df.loc[df.pixel_size_3==0,'slice_thickness']
# Axial FOV
df['axial_fov'] = df.matrix_size_3*df.pixel_size_3
# Large enough axial FOV
prev = df
df = df[df.axial_fov>500] # over 500 mm
print('Removed if axial FOV (number of slices * slice thickness) <= 500mm:')
print(get_diff_from_prev(prev=prev, curr=df))
# No missing slices, from check_axial_pixel_size.py
prev = df
df = df[~((df.n_ps2>1)&(df.n_ps3>1)&(df.n_ps4>1))]
print('Removed if missing slices:')
print(get_diff_from_prev(prev=prev, curr=df))
# ATTN correction
prev = df
df = df[df.corrections.str.contains('ATTN')]
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed if does not have ATTN correction:')
print(get_diff_from_prev(prev=prev, curr=df))
# Remove legs
prev = df
df = df[~df.description_series.str.contains('legs',case=False)]
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed if description contains \"legs\":')
print(get_diff_from_prev(prev=prev, curr=df))
# Remove legs 2 - low image values
prev = df
df = df[df.img_mean_nonzeros>=0.52]
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed if img mean nonzeros < 0.52:')
print(get_diff_from_prev(prev=prev, curr=df))
# Keep larger FOV
df['fov_max'] = df.groupby('patient_id').axial_fov.transform('max')
prev = df
df = df[df.axial_fov>0.8*df.fov_max].reset_index(drop=True).drop(['fov_max'],axis=1)
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed if axial FOV is not within 20\% of the largest FOV available for this patient:')
print(get_diff_from_prev(prev=prev, curr=df))
# Smallest pixel_size_3/slice thickness (largest resolution)
df['pixel_size_1_min'] = df.groupby('patient_id').pixel_size_1.transform('min')
prev = df
df = df[df.pixel_size_1<=df.pixel_size_1_min].reset_index(drop=True).drop(['pixel_size_1_min'],axis=1)
df['pixel_size_3_min'] = df.groupby('patient_id').pixel_size_3.transform('min')
df = df[df.pixel_size_3<=df.pixel_size_3_min].reset_index(drop=True).drop(['pixel_size_3_min'],axis=1)
df['n'] = df.groupby('patient_id').filename.transform('count')
print('Removed if the resolution is not the best available for this patient:')
print(get_diff_from_prev(prev=prev, curr=df))


df['matrix_size_3_max'] = df.groupby('patient_id').matrix_size_3.transform('max')
prev = df
df = df[df.matrix_size_3>=df.matrix_size_3_max].reset_index(drop=True).drop(['matrix_size_3_max'],axis=1)
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed if the number of slices is not the most available for this patient:')
print(get_diff_from_prev(prev=prev, curr=df))

# Image recon - keep OSEM if many
def check_contains_osem(group,F,unique_types):
    bo = np.any([fnmatch.fnmatchcase(elem2, pattern+'*') for pattern in F for elem2 in unique_types])
    if bo:
        return group[group.reconstruction.str.startswith(tuple(F))]
    else:
        return None
def filter_reconstruction(group):
    unique_types = group['reconstruction'].unique()
    if len(unique_types) > 1:
        # Prio order
        F = ['']*4
        F[0] = np.array(('OSEM3D','WB OSEM3D'))
        F[1] = np.array(('OSEM','WB OSEM'))
        F[2] = np.array(('3D IR'))
        F[3] = np.array(('PSF','WB PSF','VPFXS','QCFX'))
        run,i = 1,0
        while run:
            if i>=len(F):
                run = 0
                out = group
            else:
                out = check_contains_osem(group,F[i],unique_types)
                if out is not None:
                    run=0
            i+=1
        return out
    else:
        # Only one type — keep all
        return group
# Apply the function grouped by subject
prev = df
df = df.groupby('patient_id', group_keys=False).apply(filter_reconstruction)
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed due to reconstruction')
print(get_diff_from_prev(prev=prev, curr=df))
# Throw out weird image values
prev = df
df = df[df.img_mean_nonzeros<2]
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed img mean nonzeros >= 2')
print(get_diff_from_prev(prev=prev, curr=df))
# Keep G4 in series description
def check_contains_g4(group,keyword,unique_types):
    if any(fnmatch.fnmatchcase(desc, f"*{keyword}*") for desc in unique_types):
        return group[group['description_series'].str.contains(keyword, na=False)]
    else:
        return None
def filter_g4(group):
    unique_types = group['description_series'].unique()
    if len(unique_types) > 1:
        for keyword in ['G4', 'G5']:
            result = check_contains_g4(group, keyword, unique_types)
            if result is not None:
                return result
    return group  # No G4 or G5 found — return all

# Apply the function grouped by subject
prev = df
df = df.groupby('patient_id', group_keys=False).apply(filter_g4)
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Removed by G4/G5 description filtering')
print(get_diff_from_prev(prev=prev, curr=df))
#1/0
# Take first one of whats left
prev = df
df = df.groupby(['patient_id','scan_date'], group_keys=False).first()
df['n'] = df.groupby(['patient_id','scan_date']).filename.transform('count')
print('Filtered out because can only keep 1 scan per patient and scan date combination:')
print(get_diff_from_prev(prev=prev, curr=df))
#1/0
df = df.sort_values(by='filename').reset_index(drop=True)
df.drop(columns='n').to_csv(os.path.join(args.output_dir, 'data_scan_curated.csv'),index=0)

# Loop over rows
exit
import shutil
for i,row in df.iterrows():
    print('[{:03d}/{:03d}] {}'.format(i+1,len(df),row.filename)) 
    

    pattern = os.path.join(os.path.dirname(args.thumbnail_save_dir),'image_thumbnails_png', str(row.filepath), '*', row.filename.replace('.h5','.png'))
    patho_list = glob.glob(pattern)
    if len(patho_list) == 0:
        assert False, f"No file with pattern {pattern}"
    patho = patho_list[0]
    B = os.path.join(args.thumbnail_save_dir, str(row.filepath), row.filename.replace('.h5','.png'))
    os.makedirs(os.path.dirname(B), exist_ok=True)
    shutil.copy(patho, B)


# Test images
#tmp,img,h,ps = check_image(df[df.filepath==8371405])
#tmp,img,h,ps = check_image(df[df.filepath==20746080])
#tmp,img,h,ps = check_image(df[df.filepath==8139510])
#tmp,img,h,ps = check_image(df[df.filepath==33952839])

#i=0
#show_mips(img[i],pixelsize=ps[i],scale=4,savename=os.path.join(args.output_dir, 

#tmp,img,h,ps = check_image(df[df.filepath == df[df.n==2].iloc[0].filepath])