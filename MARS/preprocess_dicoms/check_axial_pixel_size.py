import pandas as pd
import numpy as np
import os
import h5py
import pickle
import scipy
import pdb
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import ast
import argparse
from utils import show_mips, show_grid, bbox_nd, read_image_h5, read_header_pkl, get_sample_from_df, literalize_string_of_dict_with_numpy_type


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
###############################

parser = argparse.ArgumentParser(description='Select relevant scans from all scans.')
parser.add_argument('--input_csv', type=str, default='data_scan.csv', help='Path to the input CSV file')
parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
args = parser.parse_args()

### Data
df = pd.read_csv(args.input_csv)

df['axial_fov'] = df.matrix_size_3*df.pixel_size_3

# Loop over rows
for i,row in df.iterrows():
    print('[{:04d}/{:04d}] {}'.format(i+1,len(df),row.fullfile))
    #ms = ast.literal_eval(row.pixel_size_3_str)
    ms = literalize_string_of_dict_with_numpy_type(row.pixel_size_3_str)
    keys = np.array(list(ms.keys()))
    values = np.round(np.array(list(ms.values())),decimals=3)
    fk = np.round(keys/keys.max(),decimals=3)
    fv = np.round(values/values.max(),decimals=3)
    df.loc[i,'n_ps'] = len(fk)
    df.loc[i,'n_ps2'] = len(np.unique(fk))
    df.loc[i,'n_ps3'] = (fk<0.9).sum()
    df.loc[i,'n_ps4'] = (fv>0.5).sum()
    # values2 = np.round(np.array(list(ms.values())),decimals=2)
    # keys2 = np.round(np.array(list(ms.keys())),decimals=3) 
    # if len(f)==3:
    #     pdb.set_trace()
    #df.loc[i,'n_ps2'] = len(f)
    print('   ',len(keys),keys,values,fk)
    # if (f>0.5).sum()>1:
    #     pdb.set_trace()
    #pdb.set_trace()

df.to_csv(os.path.join(args.output_dir, 'data_count_axial_pixel_size.csv'),index=0)
