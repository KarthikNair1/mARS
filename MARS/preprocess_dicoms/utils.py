import itertools
import pandas as pd
import numpy as np
import os
import h5py
import pydicom 
import glob
#import tkinter
#os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#import itertools
import pickle
import scipy
import pdb
import time
import argparse
from collections import Counter
import ast

def show_mips(img,figno=1,pixelsize=[1,1,1],scale=6,savename=[]):
    fig = plt.figure(num=figno,figsize=(2*scale, 1*scale), clear=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 2),  # creates grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for i in range(2):
        if i==0:
            IM = np.rot90(np.max(img,axis=0).squeeze(),1)
            asp = pixelsize[2]/pixelsize[1]
        elif i==1:
            IM = np.rot90(np.max(img,axis=1).squeeze(),1)
            asp = pixelsize[2]/pixelsize[0]
        #grid[i].imshow(IM, vmin=IM.min(), vmax=IM.max()*0.5, cmap=plt.get_cmap('gray'))
        grid[i].imshow(IM, vmin=0, vmax=10, cmap=plt.get_cmap('gray_r'))
        grid[i].set_aspect(asp)
        grid[i].patch.set_edgecolor('black')  
        grid[i].patch.set_linewidth(1)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    plt.subplots_adjust(wspace=.0,hspace=.0,bottom=0.01,top=0.98,left=0.01,right=0.99)
    #plt.show(block=False); plt.draw()
    if savename!=[]:
        depth = len(savename.split(os.sep))
        if depth >= 2:
            os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename+'.png', facecolor='w', edgecolor=None,format='png', transparent=False, bbox_inches='tight')
    plt.close(fig)

def show_grid(img,figno=1,pixelsize=[1,1,1], savename=[], scale=2):
    fig = plt.figure(num=figno,figsize=(3*scale, 1*scale), clear=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 3),  # creates grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )
    for i in range(3):
        if i==0:
            IM = img[int(np.round(img.shape[0]/2)),:,:].squeeze()
            asp = pixelsize[1]/pixelsize[2]
        elif i==1:
            IM = img[:,int(np.round(img.shape[1]/2)),:].squeeze()
            asp = pixelsize[0]/pixelsize[2]
        else: 
            IM = img[:,:,int(np.round(img.shape[2]/2))].squeeze()
            asp = pixelsize[0]/pixelsize[1]
        grid[i].imshow(IM, vmin=IM.min(), vmax=IM.max(), cmap=plt.get_cmap('gray'))
        grid[i].set_aspect(asp)
        grid[i].patch.set_edgecolor('black')  
        grid[i].patch.set_linewidth(1)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
    plt.subplots_adjust(wspace=.0,hspace=.0,bottom=0.01,top=0.98,left=0.01,right=0.99)
    #plt.show(block=False); plt.draw()
    if savename!=[]:
        plt.savefig(savename+'.png', facecolor='w', edgecolor=None,format='png', transparent=False, bbox_inches='tight')
    plt.close(fig)

def bbox_nd(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)

def read_image_h5(filename):
    f = h5py.File(filename, "r")
    img = f['image']
    return img
def read_header_pkl(filename):
    with open(filename, "rb") as f:
        h = pickle.load(f)
    return h
def get_sample_from_df(df, rt):
    print(df.filepath)
    img = read_image_h5(f'{rt}/images_h5/'+df.filepath+'/'+df.filename)
    h = read_header_pkl(f'{rt}/headers_pkl/'+df.filepath+'/'+df.filename.replace('.h5','.pkl'))
    ps = [df.pixel_size_1,df.pixel_size_2,df.pixel_size_3]
    return df,img,h,ps

def literalize_string_of_dict_with_numpy_type(st):
    # Given a string like '{np.float64(3.0): 619}', return the literalized dict eg {3.0: 619}
    st = st.replace('np.float64(', '')
    st = st.replace(')', '')
    return ast.literal_eval(st)