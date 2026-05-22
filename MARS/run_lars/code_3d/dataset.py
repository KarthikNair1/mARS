import numpy as np
import pandas as pd
import os
import glob
import torch
import torch.nn as nn
import pdb
import random
import itertools
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import h5py
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d, gaussian_filter
import re
import cv2 as cv
try:
    import utils
except:
    from MARS.run_lars.code_3d import utils

def get_datasets_3d(dataset_type, transform=None, norm=None, balance=False, data_path_img = None, data_path = None, 
                    postfilter_fwhm_xy_mm=None, normalize_mean=None, normalize_std=None, target_label_name = 'PET_jco',
                    postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None,
                    fixed_suv_clip=None,
                    suv_clip_threshold_min=None, suv_clip_threshold_max=None,
                    ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
                    ct_clip_threshold_low=-125, ct_clip_threshold_high=225):
    
    assert balance == False, "balance not implemented"
    df = pd.read_csv(data_path)
    
    # Balance weight
    #weight_neg_pos = [1-(df.target==0).sum()/len(df), 1-(df.target==1).sum()/len(df)]    

    if dataset_type == 'train':
        dset = dataset_3d(df, transform=transform, norm=norm, 
                          data_path_img=data_path_img, data_path=data_path, 
                          postfilter_fwhm_xy_mm=postfilter_fwhm_xy_mm,
                          normalize_mean=normalize_mean, normalize_std=normalize_std,
                          target_label_name=target_label_name,
                          postfilter_fwhm_xy_mm_min=postfilter_fwhm_xy_mm_min,
                          postfilter_fwhm_xy_mm_max=postfilter_fwhm_xy_mm_max,
                          fixed_suv_clip=fixed_suv_clip,
                          suv_clip_threshold_min=suv_clip_threshold_min,
                          suv_clip_threshold_max=suv_clip_threshold_max,
                          ct_mode=ct_mode, ct_bone_seg_dilation_erosion_pixels=ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=ct_bone_seg_dilation_alone_pixels,
                          ct_clip_threshold_low=ct_clip_threshold_low, ct_clip_threshold_high=ct_clip_threshold_high
        )
    elif dataset_type in ['val', 'test']:
        dset = dataset_3d_center(df, transform=transform, norm=norm, 
                                 data_path_img=data_path_img, data_path=data_path, 
                                 postfilter_fwhm_xy_mm=postfilter_fwhm_xy_mm,
                                 normalize_mean=normalize_mean, normalize_std=normalize_std,
                                 target_label_name=target_label_name,
                                 postfilter_fwhm_xy_mm_min=None,
                                 postfilter_fwhm_xy_mm_max=None,
                                 fixed_suv_clip=fixed_suv_clip,
                                 ct_mode=ct_mode, ct_bone_seg_dilation_erosion_pixels=ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=ct_bone_seg_dilation_alone_pixels,
                                 ct_clip_threshold_low=ct_clip_threshold_low, ct_clip_threshold_high=ct_clip_threshold_high
        )
    else:
        assert False, 'dataset_type not recognized'
    return dset

def preprocess(df):
    tmp = df.copy()
    # Normalize
    keys = []
    # pdb.set_trace()
    if 'SEX' in tmp:
        tmp['SEX'] = tmp['SEX'].fillna(tmp['SEX'].mode().values[0])
        keys = keys + ['SEX']
    if 'RACE' in tmp:
        tmp['RACE'] = tmp['RACE'].fillna(tmp['RACE'].mode().values[0])
        keys = keys + ['RACE']
    if 'AGE_EXAM' in tmp:
        tmp['AGE_EXAM'] = tmp['AGE_EXAM'].fillna(tmp['AGE_EXAM'].mean())
        keys = keys + ['AGE_EXAM']
    mu = dict()
    sd = dict()
    for key in keys:
        mu[key] = tmp[key].mean()
        sd[key] = tmp[key].std()
        tmp[key] = (tmp[key] - tmp[key].mean()) / tmp[key].std()
    return tmp,mu,sd

def get_bbox(img, ct_mode=None, fixed_bbox=None):
    if fixed_bbox is not None:
        rmin, rmax, cmin, cmax, zmin, zmax = fixed_bbox
        return img[rmin:rmax, cmin:cmax, zmin:zmax], fixed_bbox
    
    if ct_mode == 'CT' or ct_mode == 'BONE_SEG':
        # note, bone_seg only has values 0 or 1, so we shouldnt filter any of these 0s out
        rows = np.any(img >= -800, axis=1)
        cols = np.any(img >= -800, axis=0)
    elif ct_mode is None:
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
    rmin, rmax = min(np.where(rows)[0]), max(np.where(rows)[0])
    cmin, cmax = min(np.where(cols)[0]), max(np.where(cols)[0])
    zmin, zmax = min(np.where(rows)[1]), max(np.where(rows)[1])
    return img[rmin:rmax, cmin:cmax, zmin:zmax], (rmin, rmax, cmin, cmax, zmin, zmax)

def pad2square_random(image, size, is_ct):
    #out = np.zeros(size)
    out = np.full(size, -1024.0) if is_ct else np.zeros(size)

    # Sample offset
    maxr = size[0] - image.shape[0]
    maxc = size[1] - image.shape[1]
    maxz = size[2] - image.shape[2]
    offsetc = np.random.randint(0, maxc + 1) # as maxc is an attainable value, so we want it to be inclusive
    offsetr = np.random.randint(0, maxr + 1)
    offsetz = np.random.randint(0, maxz + 1)
    # Place image
    out[offsetr:offsetr+image.shape[0], offsetc:offsetc+image.shape[1], offsetz:offsetz+image.shape[2]] = image
    return out, (offsetr, offsetc, offsetz)

def pad2square_center(image, size, is_ct):
    # Place image
    out = np.full(size, -1024.0) if is_ct else np.zeros(size)

    x1 = int((size[0]-image.shape[0])/2)
    y1 = int((size[1]-image.shape[1])/2)
    z1 = int((size[2]-image.shape[2])/2)

    out[x1:x1+image.shape[0],y1:y1+image.shape[1],z1:z1+image.shape[2]] = image
    return out, (x1, y1, z1)
def pad2square_fixed(image, size, offset, is_ct):
    # offset is tuple(offsetR, offsetC, offsetZ)
    out = np.full(size, -1024.0) if is_ct else np.zeros(size)
    out[offset[0]:offset[0]+image.shape[0], offset[1]:offset[1]+image.shape[1], offset[2]:offset[2]+image.shape[2]] = image
    return out, offset

def get_image(df, transform, norm, data_path_img, data_path, fwhm_xy_mm = None, normalize_mean=None, normalize_std=None, suv_clip_threshold = None, 
              ct_mode = None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
              fixed_bbox = None, fixed_offset = None,
              ct_clip_threshold_low=-125, ct_clip_threshold_high=225):
    
    image_size = [250,250,300] 
    if ct_mode == 'CT':
        data_path_ct = data_path_img.replace('images_h5', 'images_h5_aligned_ct')
        f = h5py.File(os.path.join(data_path_ct,df.fullfile_ct), "r")
        img = f['image']
    elif ct_mode is not None and ct_mode.startswith('BONE_SEG'):
        data_path_bone_seg = data_path_img.replace('images_h5','ct_bone_segmentations')
        f = h5py.File(os.path.join(data_path_bone_seg,df.fullfile_ct_bone_segmentation), "r")
        img = f['segmentation']

        # assert that its only 0s and 1s
        assert set(np.unique(img)).issubset({0,1}), "Bone segmentation image has values other than 0 and 1"

        if ct_bone_seg_dilation_erosion_pixels is not None:
            img = img.astype(np.uint8)
            img = utils.smart_binary_dilation(img, ct_bone_seg_dilation_erosion_pixels)
            img = utils.smart_binary_erosion(img, ct_bone_seg_dilation_erosion_pixels)
            img = utils.smart_binary_dilation(img, ct_bone_seg_dilation_alone_pixels)
    else:
        f = h5py.File(os.path.join(data_path_img,df.fullfile), "r")
        img = f['image']

    # Post-filter
    if fwhm_xy_mm is not None:
        ps_xy_mm = df.pixel_size_1 #mm
        fwhm_xy_px = fwhm_xy_mm/ps_xy_mm #pixels
        sigma_xy_px = fwhm_xy_px/(2*np.sqrt(2*np.log(2)))
        pf_z_px = np.array((1,4,1))
        pf_z_px = pf_z_px/pf_z_px.sum()  # Normalize to preserve scale

        # Apply the 1D convolution along z-axis (axis=2), and gaussian along xy
        img = convolve1d(img, pf_z_px, axis=2, mode='nearest')
        img = gaussian_filter(img, sigma=(sigma_xy_px, sigma_xy_px, 0))

    # Resample to iso
    if ct_mode is not None:
        ps = np.array([df.pixel_size_1, df.pixel_size_2, df.pixel_size_3])
    else:
        ps = np.array([df.pixel_size_1, df.pixel_size_2, df.pixel_size_3])
    pixel_size_iso = [3.27,3.27,3.27]
    zoom = ps / pixel_size_iso

    img = scipy.ndimage.zoom(img, zoom, output=None, order=3 if ct_mode is None else 0 if 'BONE_SEG' in ct_mode else 1, 
                mode='constant', cval=-1024.0 if ct_mode=='CT' else 0.0, prefilter=True, grid_mode=False)
    
    # Find bbox
    if ct_mode is None:
        img[img<0.1] = 0
    img, bbox = get_bbox(img, ct_mode=ct_mode, fixed_bbox=fixed_bbox)

    # Crop if whole body
    longest_dimension = np.argmax(img.shape)
    if img.shape[longest_dimension]>300:
        if img.shape[longest_dimension]>400:
            img = np.take(img, indices = range(img.shape[longest_dimension] - 300, img.shape[longest_dimension]), axis=longest_dimension)
            #img = np.take(img, indices = range(20,320), axis=longest_dimension)
        elif img.shape[longest_dimension]>310:
            img = np.take(img, indices = range(img.shape[longest_dimension] - 300, img.shape[longest_dimension]), axis=longest_dimension)
            #img = np.take(img, indices = range(10,310), axis=longest_dimension)
        else:
            img = np.take(img, indices = range(img.shape[longest_dimension] - 300, img.shape[longest_dimension]), axis=longest_dimension)
    
    

    # Pad randomly
    if fixed_offset is not None:
        img, offset = pad2square_fixed(img, image_size, fixed_offset, is_ct=(ct_mode=='CT'))
    else:
        img, offset = pad2square_random(img, image_size, is_ct=(ct_mode=='CT'))

    # Norm
    if norm:
        if ct_mode == 'CT':
            img = clip_and_standardize_CTimage(
                img,
                clip_threshold_low=ct_clip_threshold_low,
                clip_threshold_high=ct_clip_threshold_high,
            )
        elif ct_mode is not None and 'BONE_SEG' in ct_mode:
            # no normalization, as it is binary
            pass
        else:
            img = clip_and_normalize_SUVimage(img, normalize_mean, normalize_std, suv_clip_threshold=suv_clip_threshold)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img, bbox, offset

def get_image_center(df, transform, norm, data_path_img, data_path, fwhm_xy_mm = None, normalize_mean=None, normalize_std=None, suv_clip_threshold = None,
              ct_mode = None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None, fixed_bbox = None,
              ct_clip_threshold_low=-125, ct_clip_threshold_high=225):
    image_size = [250,250,300] # max 213x213x300
    if ct_mode == 'CT':
        data_path_ct = data_path_img.replace('images_h5', 'images_h5_aligned_ct')
        f = h5py.File(os.path.join(data_path_ct,df.fullfile_ct), "r")
        img = f['image']
    elif ct_mode is not None and ct_mode.startswith('BONE_SEG'):
        data_path_bone_seg = data_path_img.replace('images_h5','ct_bone_segmentations')
        f = h5py.File(os.path.join(data_path_bone_seg,df.fullfile_ct_bone_segmentation), "r")
        img = f['segmentation']

        # assert that its only 0s and 1s
        assert set(np.unique(img)).issubset({0,1}), "Bone segmentation image has values other than 0 and 1"

        if ct_bone_seg_dilation_erosion_pixels is not None:
            img = img.astype(np.uint8)
            img = utils.smart_binary_dilation(img, ct_bone_seg_dilation_erosion_pixels)
            img = utils.smart_binary_erosion(img, ct_bone_seg_dilation_erosion_pixels)
            img = utils.smart_binary_dilation(img, ct_bone_seg_dilation_alone_pixels)
    else:
        f = h5py.File(os.path.join(data_path_img,df.fullfile), "r")
        img = f['image']

    # Post-filter
    if fwhm_xy_mm is not None:
        ps_xy_mm = df.pixel_size_1 #mm
        fwhm_xy_px = fwhm_xy_mm/ps_xy_mm #pixels
        sigma_xy_px = fwhm_xy_px/(2*np.sqrt(2*np.log(2)))
        pf_z_px = np.array((1,4,1))
        pf_z_px = pf_z_px/pf_z_px.sum()  # Normalize to preserve scale

        # Apply the 1D convolution along z-axis (axis=2), and gaussian along xy
        img = convolve1d(img, pf_z_px, axis=2, mode='nearest')
        img = gaussian_filter(img, sigma=(sigma_xy_px, sigma_xy_px, 0))

    # Resample to iso
    if ct_mode is not None:
        ps = np.array([df.pixel_size_1, df.pixel_size_2, df.pixel_size_3])
    else:
        ps = np.array([df.pixel_size_1, df.pixel_size_2, df.pixel_size_3])
    pixel_size_iso = [3.27,3.27,3.27]
    zoom = ps / pixel_size_iso

    img = scipy.ndimage.zoom(img, zoom, output=None, order=3 if ct_mode is None else 0 if 'BONE_SEG' in ct_mode else 1,
                mode='constant', cval=-1024.0 if ct_mode=='CT' else 0.0, prefilter=True, grid_mode=False)
    # Find bbox
    if ct_mode is None:
        img[img<0.1] = 0
    img, bbox = get_bbox(img, ct_mode=ct_mode, fixed_bbox=fixed_bbox)

    # Crop if whole body
    longest_dimension = np.argmax(img.shape)
    if img.shape[longest_dimension]>300:
        if img.shape[longest_dimension]>400:
            img = np.take(img, indices = range(img.shape[longest_dimension] - 300, img.shape[longest_dimension]), axis=longest_dimension)
        elif img.shape[longest_dimension]>310:
            img = np.take(img, indices = range(img.shape[longest_dimension] - 300, img.shape[longest_dimension]), axis=longest_dimension)
        else:
            img = np.take(img, indices = range(img.shape[longest_dimension] - 300, img.shape[longest_dimension]), axis=longest_dimension)
    img, offset = pad2square_center(img, image_size, is_ct = (ct_mode=='CT'))

    # Norm
    if norm:
        if ct_mode == 'CT':
            img = clip_and_standardize_CTimage(
                img,
                clip_threshold_low=ct_clip_threshold_low,
                clip_threshold_high=ct_clip_threshold_high,
            )
        elif ct_mode is not None and 'BONE_SEG' in ct_mode:
            # no normalization, as it is binary
            pass
        else:
            img = clip_and_normalize_SUVimage(img, normalize_mean, normalize_std, suv_clip_threshold=suv_clip_threshold)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img, bbox, offset

def get_image_sizes(size):
    image_size = [250,250,size]
    img = torch.zeros(image_size).unsqueeze(0)
    return img

def clip_and_normalize_SUVimage(img, normalize_mean=None, normalize_std=None, suv_clip_threshold=None):
    if (normalize_mean is not None) and (normalize_std is not None):
        mu = normalize_mean
        std = normalize_std
    else:
        mu  = 2.13
        std = 3.39
    #q   = 30.00
    if suv_clip_threshold is not None:
        q = suv_clip_threshold
    else:
        q   = 60.00
    
    img = np.clip(img,0.,q)

    return (img-mu)/std

def clip_and_standardize_CTimage(img, clip_threshold_low = -125, clip_threshold_high = 225): # these thresholds are per the LARS lancet paper (described in supplemental)
    img = np.clip(img, clip_threshold_low, clip_threshold_high)

    # standardize to [0,1]
    img = (img - clip_threshold_low) / (clip_threshold_high - clip_threshold_low)
    return img
   
class dataset_3d(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, norm=False, 
                data_path_img=None, data_path=None, 
                postfilter_fwhm_xy_mm=None,
                normalize_mean = None, normalize_std=None,
                target_label_name = 'PET_jco',
                postfilter_fwhm_xy_mm_min=None,
                postfilter_fwhm_xy_mm_max=None,
                fixed_suv_clip=None,
                suv_clip_threshold_min=None,
                suv_clip_threshold_max=None,
                ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
                ct_clip_threshold_low=-125, ct_clip_threshold_high=225):
        
        self.df = df.copy()
        self.transform = transform
        self.norm = norm
        self.data_path_img = data_path_img
        self.data_path = data_path
        self.postfilter_fwhm_xy_mm = postfilter_fwhm_xy_mm
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.target_label_name = target_label_name
        self.postfilter_fwhm_xy_mm_min = postfilter_fwhm_xy_mm_min
        self.postfilter_fwhm_xy_mm_max = postfilter_fwhm_xy_mm_max
        self.fixed_suv_clip = fixed_suv_clip
        self.suv_clip_threshold_min = suv_clip_threshold_min
        self.suv_clip_threshold_max = suv_clip_threshold_max
        self.ct_clip_threshold_low = ct_clip_threshold_low
        self.ct_clip_threshold_high = ct_clip_threshold_high

        if ct_mode is not None:
            assert ct_mode in ['CT', 'BONE_SEG', 'BONE_SEG_MASK_MULTIPLY', 'BONE_SEG_MASK_MULTIPLY_ONLY']
        self.ct_mode = ct_mode
        self.ct_bone_seg_dilation_erosion_pixels = ct_bone_seg_dilation_erosion_pixels
        self.ct_bone_seg_dilation_alone_pixels = ct_bone_seg_dilation_alone_pixels

    def __getitem__(self, index):
        tmp = self.df.iloc[index]

        if self.postfilter_fwhm_xy_mm is not None:
            # correct for postfiltering
            if self.postfilter_fwhm_xy_mm_min is None:
                desired_kernel = self.postfilter_fwhm_xy_mm # default should be 6.4
            else:
                if np.random.choice([0,1]) == 1:
                    desired_kernel = np.random.uniform(self.postfilter_fwhm_xy_mm_min, self.postfilter_fwhm_xy_mm_max)
                else:
                    desired_kernel = self.postfilter_fwhm_xy_mm
            
            recon_postfilter_mm = get_recon_postfilter_mm(tmp)
            fwhm_xy_mm = np.sqrt( np.clip(desired_kernel**2 - recon_postfilter_mm**2, a_min=0, a_max=None) )
        else:
            fwhm_xy_mm = None

        if self.suv_clip_threshold_min is not None:
            suv_clip_threshold = np.random.uniform(self.suv_clip_threshold_min, self.suv_clip_threshold_max)
        elif self.fixed_suv_clip is not None:
            suv_clip_threshold = self.fixed_suv_clip
        else:
            suv_clip_threshold = None
        
        # Read image
        img, bbox, offset = get_image(tmp, transform=self.transform, norm=self.norm, 
                        data_path_img=self.data_path_img, data_path=self.data_path, 
                        fwhm_xy_mm=fwhm_xy_mm, 
                        normalize_mean=self.normalize_mean, normalize_std=self.normalize_std,
                        suv_clip_threshold=suv_clip_threshold, ct_mode = None,
                        ct_clip_threshold_low=self.ct_clip_threshold_low, ct_clip_threshold_high=self.ct_clip_threshold_high)
        
        # if we are stacking the CT, read and stack
        if self.ct_mode is not None:
            ct_img, _, _ = get_image(tmp, transform=self.transform, norm=self.norm, 
                            data_path_img=self.data_path_img, data_path=self.data_path, 
                            fwhm_xy_mm=None, 
                            normalize_mean=None, normalize_std=None,
                            suv_clip_threshold=None, ct_mode = self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels,
                            ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels,
                            fixed_bbox=bbox, fixed_offset=offset,
                            ct_clip_threshold_low=self.ct_clip_threshold_low, ct_clip_threshold_high=self.ct_clip_threshold_high)

            if self.ct_mode.startswith('BONE_SEG_MASK_MULTIPLY'):
                ct_img = ct_img * img
                
            if 'ONLY' in self.ct_mode:
                img = ct_img
            else:
                img = torch.cat( (img, ct_img), dim=0 )  # concatenate along channel dimension
        
        return img, tmp[self.target_label_name]
    
    def __len__(self):
        return len(self.df)

class dataset_3d_center(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, norm=False, 
                data_path_img=None, data_path=None, 
                postfilter_fwhm_xy_mm=None,
                normalize_mean = None, normalize_std=None,
                target_label_name = 'PET_jco',
                postfilter_fwhm_xy_mm_min=None,
                postfilter_fwhm_xy_mm_max=None,
                fixed_suv_clip=None,
                ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
                ct_clip_threshold_low=-125, ct_clip_threshold_high=225):
        self.df = df.copy()
        self.transform = transform
        self.norm = norm
        self.data_path_img = data_path_img
        self.data_path = data_path
        self.postfilter_fwhm_xy_mm = postfilter_fwhm_xy_mm
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std 
        self.target_label_name = target_label_name
        self.postfilter_fwhm_xy_mm_min = postfilter_fwhm_xy_mm_min
        self.postfilter_fwhm_xy_mm_max = postfilter_fwhm_xy_mm_max
        self.fixed_suv_clip = fixed_suv_clip
        self.ct_clip_threshold_low = ct_clip_threshold_low
        self.ct_clip_threshold_high = ct_clip_threshold_high
        
        if ct_mode is not None:
            assert ct_mode in ['CT', 'BONE_SEG', 'BONE_SEG_MASK_MULTIPLY', 'BONE_SEG_MASK_MULTIPLY_ONLY']
        self.ct_mode = ct_mode
        self.ct_bone_seg_dilation_erosion_pixels = ct_bone_seg_dilation_erosion_pixels
        self.ct_bone_seg_dilation_alone_pixels = ct_bone_seg_dilation_alone_pixels

    def __getitem__(self, index): 
        tmp = self.df.iloc[index]

        if self.postfilter_fwhm_xy_mm is not None:
            # correct for postfiltering
            desired_kernel = self.postfilter_fwhm_xy_mm # default should be 6.4
            recon_postfilter_mm = get_recon_postfilter_mm(tmp)
            fwhm_xy_mm = np.sqrt( np.clip(desired_kernel**2 - recon_postfilter_mm**2, a_min=0, a_max=None) )
        else:
            fwhm_xy_mm = None

        if self.fixed_suv_clip is not None:
            suv_clip_threshold = self.fixed_suv_clip
        else:
            suv_clip_threshold = None
        # Read image
        img, bbox, _ = get_image_center(tmp, transform=self.transform, norm=self.norm, 
                        data_path_img=self.data_path_img, data_path=self.data_path, 
                        fwhm_xy_mm=fwhm_xy_mm, 
                        normalize_mean=self.normalize_mean, normalize_std=self.normalize_std,
                        suv_clip_threshold=suv_clip_threshold, ct_mode = None,
                        ct_clip_threshold_low=self.ct_clip_threshold_low, ct_clip_threshold_high=self.ct_clip_threshold_high)
        # if we are stacking the CT, read and stack
        if self.ct_mode is not None:
            ct_img, _, _ = get_image_center(tmp, transform=self.transform, norm=self.norm, 
                            data_path_img=self.data_path_img, data_path=self.data_path, 
                            fwhm_xy_mm=None, 
                            normalize_mean=None, normalize_std=None,
                            suv_clip_threshold=None, ct_mode = self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels,
                            ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels,
                            fixed_bbox=bbox,
                            ct_clip_threshold_low=self.ct_clip_threshold_low, ct_clip_threshold_high=self.ct_clip_threshold_high)
            
            if self.ct_mode.startswith('BONE_SEG_MASK_MULTIPLY'):
                ct_img = ct_img * img
                
            if 'ONLY' in self.ct_mode:
                img = ct_img
            else:
                img = torch.cat( (img, ct_img), dim=0 )  # concatenate along channel dimension
        
        #return img, tmp.target
        return img, tmp[self.target_label_name]
    
    def __len__(self):
        return len(self.df)

class RandomFlip(object):
    """Randomly flip left-right, ant-post.
    """
    def __call__(self, image):
        # Random flip
        fliplr = random.choice((0,1))
        flipap = random.choice((0,1))
        if fliplr==1:
            image   = image[range(image.shape[0]-1,-1,-1),:,:]
        if flipap==1:
            image   = image[:,range(image.shape[1]-1,-1,-1),:]       
        return image


def get_recon_postfilter_mm(tmp):
    if 'recon_postfilter_mm' in tmp:
        recon_postfilter_mm = tmp.recon_postfilter_mm
        if pd.isna(recon_postfilter_mm):
            return 4.0 # default 4 mm, average
        return recon_postfilter_mm
    elif 'convolution_kernel' in tmp:
        # set value to the first float in the string tmp['convolution_kernel'] using regex
        if pd.isna(tmp.convolution_kernel):
            return 4.0

        match = re.search(r"\d+\.\d+", tmp.convolution_kernel)
        if match:
            recon_postfilter_mm = float(match.group())
            return recon_postfilter_mm
        else:
            return 0 # likely an all-pass
    else:
        return 4.0
