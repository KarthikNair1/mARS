import dataset
import utils
import os
import torch
import numpy as np
import pandas as pd
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

###############################
def show_mips(cor,sag,figno=1,pixelsize=[1,1,1],scale=6,savename=[]):
    fig = plt.figure(num=figno,figsize=(2*scale, 1*scale), clear=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 2),  # creates grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for i in range(2):
        if i==0:
            IM = cor
            asp = pixelsize[2]/pixelsize[1]
        elif i==1:
            IM = sag
            asp = pixelsize[2]/pixelsize[0]
        #grid[i].imshow(IM, vmin=IM.min(), vmax=IM.max()*0.5, cmap=plt.get_cmap('gray'))
        grid[i].imshow(IM, vmin=0, vmax=10, cmap=plt.get_cmap('gray_r'))
        grid[i].set_aspect(asp)
        grid[i].patch.set_edgecolor('black')  
        grid[i].patch.set_linewidth(1)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    plt.subplots_adjust(wspace=.0,hspace=.0,bottom=0.01,top=0.98,left=0.01,right=0.99)
    if savename!=[]:
        plt.savefig(savename+'.png', facecolor='w', edgecolor=None,format='png', transparent=False, bbox_inches='tight')
    plt.close(fig)
###############################

dset = dataset.get_datasets(None,False)
df = dset.df
df = df[df.mipview=='cor']

for i,row in df.iterrows():
    print('[{:03d}}/{:03d}] {}'.format(i+1,len(df),row.filename))
    cor = dataset.get_image_center(row, 'cor', None, False).squeeze()
    sag = dataset.get_image_center(row, 'sag', None, False).squeeze()
    show_mips(cor,sag,savename='pngs/'+row.filename.replace('.h5',''))

print('Done.')