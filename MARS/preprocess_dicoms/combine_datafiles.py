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
import argparse
from utils import show_mips, show_grid, bbox_nd, read_image_h5, read_header_pkl, get_sample_from_df


plt.rcParams.update({'figure.max_open_warning': 0})
parser = argparse.ArgumentParser(description='Combine DICOM Datafiles')
parser.add_argument('--datafile_pattern', default=None, type=str, help='Pattern for data_scan csvs, eg /gpfs/data/asdf/data_scan_*.csv')
parser.add_argument('--output_file', default='', type=str, help='Output file path for combined data')
args = parser.parse_args()
### Read all files and save as single file
files = glob.glob(args.datafile_pattern)
df = pd.DataFrame()
for i,f in enumerate(files):
    df = pd.concat([df,pd.read_csv(f)])
    print(f)
df['filepath'] = df['filepath'].astype(str)

print('Done concatenating {} files'.format(i+1))
df = df.drop_duplicates().sort_values(by='filepath').reset_index(drop=True)
df.to_csv(args.output_file,index=0)

tmp,img,h,ps = get_sample_from_df(df.sample(1).iloc[0], rt='/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars')
show_mips(img,pixelsize=ps,scale=4,savename='test')
show_grid(img,pixelsize=ps,scale=4,savename='test_ffs')