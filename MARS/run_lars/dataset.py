import numpy as np
import pandas as pd
import os
import glob
import scipy
import torch
import torch.nn as nn
import torchvision
import pdb
import random
import itertools
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from scipy import interpolate
import h5py
from scipy.ndimage import convolve1d, gaussian_filter
import re
try:
    import utils
except:
    from MARS.run_lars import utils

def get_datasets(dataset_type, norm=None,transform=None, data_path_img=None, data_path=None, subset_ids=None,
                postfilter_fwhm_xy_mm=None, normalize_mean=None, normalize_std=None, target_label_name = 'PET_jco',
                postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None,
                suv_clip_threshold_min=None, suv_clip_threshold_max=None, combine_mipviews = None,
                ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
                auxiliary_label_names=None):
    df = pd.read_csv(data_path)

    if combine_mipviews is None:
        df = pd.concat([df,df]).reset_index(drop=True)
        df['mipview'] = ['cor']*int(len(df)/2) + ['sag']*int(len(df)/2)
    
    assert dataset_type in ['train','val','test']

    # subset df using subset_ids
    if subset_ids is not None:
        df = df[df['accession'].isin(subset_ids)].reset_index(drop=True)

    if dataset_type == 'train':
        dset = dataset(df, data_path_img=data_path_img, data_path=data_path, 
                       norm=norm, transform=transform, postfilter_fwhm_xy_mm=postfilter_fwhm_xy_mm,
                       normalize_mean=normalize_mean, normalize_std=normalize_std,
                       target_label_name=target_label_name,
                       postfilter_fwhm_xy_mm_min = postfilter_fwhm_xy_mm_min,
                       postfilter_fwhm_xy_mm_max = postfilter_fwhm_xy_mm_max,
                       suv_clip_threshold_min = suv_clip_threshold_min,
                       suv_clip_threshold_max = suv_clip_threshold_max,
                       combine_mipviews = combine_mipviews,
                       ct_mode = ct_mode,
                       ct_bone_seg_dilation_erosion_pixels = ct_bone_seg_dilation_erosion_pixels,
                       ct_bone_seg_dilation_alone_pixels = ct_bone_seg_dilation_alone_pixels,
                       auxiliary_label_names = auxiliary_label_names)
        
    elif dataset_type in ['val', 'test']:
        dset = dataset_center(df, data_path_img=data_path_img, data_path=data_path, 
                             norm=norm, transform=transform, postfilter_fwhm_xy_mm=postfilter_fwhm_xy_mm,
                             normalize_mean=normalize_mean, normalize_std=normalize_std,
                             target_label_name=target_label_name,
                             combine_mipviews = combine_mipviews,
                             ct_mode = ct_mode,
                             ct_bone_seg_dilation_erosion_pixels = ct_bone_seg_dilation_erosion_pixels,
                             ct_bone_seg_dilation_alone_pixels = ct_bone_seg_dilation_alone_pixels,
                             auxiliary_label_names = auxiliary_label_names)
    
    return dset
  
class dataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path_img, data_path, norm=False, transform=None, postfilter_fwhm_xy_mm=None, normalize_mean=None, normalize_std=None, target_label_name = 'PET_jco',
                 postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None, suv_clip_threshold_min=None, suv_clip_threshold_max=None, combine_mipviews=None,
                 ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
                 auxiliary_label_names=None):
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
        self.suv_clip_threshold_min = suv_clip_threshold_min
        self.suv_clip_threshold_max = suv_clip_threshold_max
        self.combine_mipviews = combine_mipviews
        self.auxiliary_label_names = auxiliary_label_names or []

        if ct_mode is not None:
            assert ct_mode in ['BONE_SEG_MASK_MULTIPLY']
        self.ct_mode = ct_mode
        self.ct_bone_seg_dilation_erosion_pixels = ct_bone_seg_dilation_erosion_pixels
        self.ct_bone_seg_dilation_alone_pixels = ct_bone_seg_dilation_alone_pixels

    def __getitem__(self, index): 

        tmp = self.df.iloc[index]
        
        if self.postfilter_fwhm_xy_mm is not None:
            # make sure to correct for postfiltering kernel size differences
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
        else:
            suv_clip_threshold = None
        
        # Read image
        if self.combine_mipviews == 'stack' or self.combine_mipviews == 'feature_concatenation':
            cor = get_image(tmp, 'cor', self.norm, self.transform, self.data_path_img, self.data_path, fwhm_xy_mm, self.normalize_mean, self.normalize_std, suv_clip_threshold, ct_mode=self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels)
            sag = get_image(tmp, 'sag', self.norm, self.transform, self.data_path_img, self.data_path, fwhm_xy_mm, self.normalize_mean, self.normalize_std, suv_clip_threshold, ct_mode=self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels)
            img = torch.cat([cor, sag], dim=0)
        else:
            img = get_image(tmp, tmp.mipview, self.norm, self.transform, self.data_path_img, self.data_path, fwhm_xy_mm, self.normalize_mean, self.normalize_std, suv_clip_threshold, ct_mode=self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels)
            
        main_label = tmp[self.target_label_name]

        if len(self.auxiliary_label_names) == 0:
            return img, main_label

        aux_labels = []
        for label_name in self.auxiliary_label_names:
            aux_val = tmp.get(label_name, np.nan)
            if pd.isna(aux_val):
                aux_val = np.nan
            aux_labels.append(aux_val)

        targets = {
            'main': main_label,
            'aux': np.array(aux_labels, dtype=np.float32)
        }
        return img, targets
    
    def __len__(self):
        return len(self.df)

class dataset_center(torch.utils.data.Dataset):
    def __init__(self, df, data_path_img, data_path, norm=False, transform=None, postfilter_fwhm_xy_mm=None, normalize_mean=None, normalize_std=None, target_label_name = 'PET_jco',
                    combine_mipviews = None,
                    ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None,
                    auxiliary_label_names=None):
        self.df = df.copy()
        self.transform = transform
        self.norm = norm
        self.data_path_img = data_path_img
        self.data_path = data_path
        self.postfilter_fwhm_xy_mm = postfilter_fwhm_xy_mm
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.target_label_name = target_label_name
        self.combine_mipviews = combine_mipviews
        self.auxiliary_label_names = auxiliary_label_names or []

        if ct_mode is not None:
            assert ct_mode in ['BONE_SEG_MASK_MULTIPLY']
        self.ct_mode = ct_mode
        self.ct_bone_seg_dilation_erosion_pixels = ct_bone_seg_dilation_erosion_pixels
        self.ct_bone_seg_dilation_alone_pixels = ct_bone_seg_dilation_alone_pixels

    def __getitem__(self, index): 
        tmp = self.df.iloc[index]

        if self.postfilter_fwhm_xy_mm is not None:
            # make sure to correct for postfiltering kernel size differences
            desired_kernel = self.postfilter_fwhm_xy_mm # default should be 6.4
            recon_postfilter_mm = get_recon_postfilter_mm(tmp)
            fwhm_xy_mm = np.sqrt( np.clip(desired_kernel**2 - recon_postfilter_mm**2, a_min=0, a_max=None) )
        else:
            fwhm_xy_mm = None    
        
        # Read image
        if self.combine_mipviews == 'stack' or self.combine_mipviews == 'feature_concatenation':
            cor = get_image_center(tmp, 'cor', self.norm, self.transform, self.data_path_img, self.data_path, fwhm_xy_mm, self.normalize_mean, self.normalize_std, ct_mode=self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels)
            sag = get_image_center(tmp, 'sag', self.norm, self.transform, self.data_path_img, self.data_path, fwhm_xy_mm, self.normalize_mean, self.normalize_std, ct_mode=self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels)
            img = torch.cat([cor, sag], dim=0)
        else:
            img = get_image_center(tmp, tmp.mipview, self.norm, self.transform, self.data_path_img, self.data_path, fwhm_xy_mm, self.normalize_mean, self.normalize_std, ct_mode=self.ct_mode, ct_bone_seg_dilation_erosion_pixels=self.ct_bone_seg_dilation_erosion_pixels, ct_bone_seg_dilation_alone_pixels=self.ct_bone_seg_dilation_alone_pixels)
        main_label = tmp[self.target_label_name]

        if len(self.auxiliary_label_names) == 0:
            return img, main_label

        aux_labels = []
        for label_name in self.auxiliary_label_names:
            aux_val = tmp.get(label_name, np.nan)
            if pd.isna(aux_val):
                aux_val = np.nan
            aux_labels.append(aux_val)

        targets = {
            'main': main_label,
            'aux': np.array(aux_labels, dtype=np.float32)
        }
        return img, targets
    
    def __len__(self):
        return len(self.df)

def get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    if not np.any(img):
        # everything is 0
        print('Blank image detected!')
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax, cmin:cmax]

def pad2square_random(image, size):
    out = np.zeros((size,size))
    # Sample offset
    maxr = size - image.shape[0]
    maxc = size - image.shape[1]
    offsetc = np.random.randint(0, maxc + 1) # as maxc is an attainable value, so we want it to be inclusive
    offsetr = np.random.randint(0, maxr + 1)
    # Place image
    out[offsetr:offsetr+image.shape[0], offsetc:offsetc+image.shape[1]] = image
    return out

def pad2square_center(image, size):
    # Place image
    out = np.zeros((size,size))
    out[int((size-image.shape[0])/2):int((size-image.shape[0])/2)+image.shape[0],int((size-image.shape[1])/2):int((size-image.shape[1])/2)+image.shape[1]] = image
    return out

def padwidth_center(image, size):
    # Place image
    out = np.zeros((image.shape[0],size))
    out[:,int((size-image.shape[1])/2):int((size-image.shape[1])/2)+image.shape[1]] = image
    return out

def padheight_center(image, size):
    # Place image
    out = np.zeros((size,image.shape[1]))
    out[int((size-image.shape[0])/2):int((size-image.shape[0])/2)+image.shape[0],:] = image
    return out

def get_image(df, mipview, norm, transform, data_path_img, data_path, fwhm_xy_mm = None, normalize_mean = None, normalize_std = None, suv_clip_threshold = None,
              ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None):
    image_size = 310
    f = h5py.File(os.path.join(data_path_img,df.fullfile), "r")
    img = f['image'][:]

    # if using CT, load it in
    if ct_mode is not None:
        assert ct_mode == 'BONE_SEG_MASK_MULTIPLY'
        data_path_bone_seg = data_path_img.replace('images_h5','ct_bone_segmentations')
        f = h5py.File(os.path.join(data_path_bone_seg,df.fullfile_ct_bone_segmentation), "r")
        ct_img = f['segmentation'][:]

        assert img.shape == ct_img.shape, f"CT and PET image shapes do not match!: PET shape {img.shape}, CT shape {ct_img.shape}"
        # apply dilation / erosion
        assert set(np.unique(ct_img)).issubset({0,1}), "Bone segmentation image has values other than 0 and 1"
        ct_img = ct_img.astype(np.uint8)
        if ct_bone_seg_dilation_erosion_pixels is not None:
            ct_img = utils.smart_binary_dilation(ct_img, ct_bone_seg_dilation_erosion_pixels)
            ct_img = utils.smart_binary_erosion(ct_img, ct_bone_seg_dilation_erosion_pixels)
        if ct_bone_seg_dilation_alone_pixels is not None:
            ct_img = utils.smart_binary_dilation(ct_img, ct_bone_seg_dilation_alone_pixels)        

        # dot product
        img = img * ct_img

    if fwhm_xy_mm is not None:
        # Post-filter
        ps_xy_mm = df.pixel_size_1 #mm
        fwhm_xy_px = fwhm_xy_mm/ps_xy_mm #pixels
        sigma_xy_px = fwhm_xy_px/(2*np.sqrt(2*np.log(2)))
        pf_z_px = np.array((1,4,1))
        pf_z_px = pf_z_px/pf_z_px.sum()  # Normalize to preserve scale

        # Apply the 1D convolution along z-axis (axis=2), and gaussian along xy
        img = convolve1d(img, pf_z_px, axis=2, mode='nearest')
        img = gaussian_filter(img, sigma=(sigma_xy_px, sigma_xy_px, 0))

    # MIP view
    if mipview=='cor':
        img = np.rot90(np.max(img,axis=0).squeeze(),1) # so using second and third axes, and then flips order of axes
        ps = np.array([df.pixel_size_3, df.pixel_size_2])
    elif mipview =='sag':
        img = np.rot90(np.max(img,axis=1).squeeze(),1) # so using first and third axes
        ps = np.array([df.pixel_size_3, df.pixel_size_1])
    # Resample to iso
    #import matplotlib.pyplot as plt
    #plt.imshow(img,vmin=0,vmax=10,cmap='gray_r')
    #plt.show(block=False); plt.draw()
    #plt.savefig('test.png', facecolor='w', edgecolor=None,format='png', transparent=False, bbox_inches='tight')
    #pdb.set_trace()
    pixel_size_iso = [3.27,3.27]
    zoom = ps / pixel_size_iso
    img = scipy.ndimage.zoom(img, zoom, output=None, order=1, 
                mode='constant', cval=0.0, prefilter=True, grid_mode=False)
    # Find bbox
    img[img<0.1] = 0
    img = get_bbox(img)
    # Crop if whole body
    if img.shape[0]>300:
        if img.shape[0]>400:
            img = img[20:320,:]
        elif img.shape[0]>310:
            img = img[10:310,:]
        else:
            img = img[0:300,:]
    # Pad randomly
    img = pad2square_random(img, image_size)
    # Norm
    if norm:
        img = clip_and_normalize_SUVimage(img, normalize_mean, normalize_std, suv_clip_threshold)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img
    
def get_image_center(df, mipview, norm, transform, data_path_img, data_path, fwhm_xy_mm=None, normalize_mean=None, normalize_std=None,
                     ct_mode=None, ct_bone_seg_dilation_erosion_pixels=None, ct_bone_seg_dilation_alone_pixels=None):
    image_size = 310
    f = h5py.File(os.path.join(data_path_img,df.fullfile), "r")
    img = f['image'][:]

    # if using CT, load it in
    if ct_mode is not None:
        assert ct_mode == 'BONE_SEG_MASK_MULTIPLY'
        data_path_bone_seg = data_path_img.replace('images_h5','ct_bone_segmentations')
        f = h5py.File(os.path.join(data_path_bone_seg,df.fullfile_ct_bone_segmentation), "r")
        ct_img = f['segmentation'][:]

        assert img.shape == ct_img.shape, f"CT and PET image shapes do not match!: PET shape {img.shape}, CT shape {ct_img.shape}"
        # apply dilation / erosion
        assert set(np.unique(ct_img)).issubset({0,1}), "Bone segmentation image has values other than 0 and 1"
        ct_img = ct_img.astype(np.uint8)
        if ct_bone_seg_dilation_erosion_pixels is not None:
            ct_img = utils.smart_binary_dilation(ct_img, ct_bone_seg_dilation_erosion_pixels)
            ct_img = utils.smart_binary_erosion(ct_img, ct_bone_seg_dilation_erosion_pixels)
        if ct_bone_seg_dilation_alone_pixels is not None:
            ct_img = utils.smart_binary_dilation(ct_img, ct_bone_seg_dilation_alone_pixels)

        # dot product
        img = img * ct_img

    if fwhm_xy_mm is not None:
        # Post-filter
        ps_xy_mm = df.pixel_size_1 #mm
        fwhm_xy_px = fwhm_xy_mm/ps_xy_mm #pixels
        sigma_xy_px = fwhm_xy_px/(2*np.sqrt(2*np.log(2)))
        pf_z_px = np.array((1,4,1))
        pf_z_px = pf_z_px/pf_z_px.sum()  # Normalize to preserve scale

        # Apply the 1D convolution along z-axis (axis=2), and gaussian along xy
        img = convolve1d(img, pf_z_px, axis=2, mode='nearest')
        img = gaussian_filter(img, sigma=(sigma_xy_px, sigma_xy_px, 0))


    # MIP view
    if mipview=='cor':
        img = np.rot90(np.max(img,axis=0).squeeze(),1)
        ps = np.array([df.pixel_size_3, df.pixel_size_2])
    elif mipview =='sag':
        img = np.rot90(np.max(img,axis=1).squeeze(),1)
        ps = np.array([df.pixel_size_3, df.pixel_size_1])
    # Resample to iso
    #import matplotlib.pyplot as plt
    #plt.imshow(img,vmin=0,vmax=10,cmap='gray_r')
    #plt.show(block=False); plt.draw()
    #plt.savefig('test.png', facecolor='w', edgecolor=None,format='png', transparent=False, bbox_inches='tight')
    #pdb.set_trace()
    pixel_size_iso = [3.27,3.27]
    zoom = ps / pixel_size_iso
    img = scipy.ndimage.zoom(img, zoom, output=None, order=1, 
                mode='constant', cval=0.0, prefilter=True, grid_mode=False)
    # Find bbox
    img[img<0.1] = 0
    #pdb.set_trace()
    img = get_bbox(img)
    # Crop if whole body
    if img.shape[0]>300:
        if img.shape[0]>400:
            img = img[20:320,:]
        elif img.shape[0]>310:
            img = img[10:310,:]
        else:
            img = img[0:300,:]
    # Pad randomly
    img = pad2square_center(img, image_size)
    # Norm
    if norm:
        img = clip_and_normalize_SUVimage(img, normalize_mean, normalize_std)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img


def get_image_center_forpaper(df, mipview, norm, transform):
    image_width,image_height = 260,260
    f = h5py.File(df.filename, "r")
    img = f['image']
    # Find bbox
    img = get_bbox(img)
    # Pad
    img = padwidth_center(img, image_width)
    img = padheight_center(img, img.shape[0]+8)
    # resample
    H, W = img.shape[:2]
    new_H, new_W = (image_height,image_width)
    xrange = lambda x: np.linspace(0, 1, x)
    f = interpolate.interp2d(xrange(W), xrange(H), img, kind="linear")
    img = f(xrange(new_W), xrange(new_H))
    # Norm
    if norm:
        img = clip_and_normalize_SUVimage(img)
    # Make Tensors
    img = torch.FloatTensor(img).unsqueeze(0)
    if transform is not None:
        img = [transform(x) for x in img]
        img = torch.stack(img)
    return img

def clip_and_normalize_SUVimage(img, normalize_mean=None, normalize_std=None, clip_threshold = 30.00):
    #2D images (sag + cor): 
    if (normalize_mean is not None) and (normalize_std is not None):
        mu = normalize_mean
        std = normalize_std
    else:
        mu  = 2.13
        std = 3.39
    if clip_threshold is None:
        clip_threshold = 30.00
    q   = clip_threshold
    img = np.clip(img,0.,q)
    return (img-mu)/std

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

class RandomFlip(object):
    """Randomly flip the 2D image.
    """
    def __call__(self, image):
        # Random flip: none, 0=vertical, 1=horizontal
        flip = random.choice((None,0,1))
        if flip is not None:
            if flip==0:
                image   = image[range(image.shape[flip]-1,-1,-1),:]
            elif flip==1:
                image   = image[:,range(image.shape[flip]-1,-1,-1)]       
        return image

class RandomFlipLeftRight(object):
    """Randomly flip all channels of the 2D image.
    """
    def __call__(self, image):
        # Random flip: none, 0=vertical, 1=horizontal
        flip = random.choice((None,1))
        if flip is not None:
            image = image[:,range(image.shape[1]-1,-1,-1)]
        return image
        
class RandomRot90(object):
    """Randomly rotate the 2D image by n*90 degrees.
    """
    def __call__(self, image):
        # Random  90 rotation
        rot = random.randint(0,3)
        if rot != 0:
            image = torch.rot90(image, rot, (0,1))
        return image

class RandomScale(object):
    """Randomly scale the 2D image.
    """
    def __call__(self, image):
        scale = np.random.uniform(low=0.85, high=1.15, size=1)
        image = image*scale[0]
        return image

class RandomNoise(object):
    """Randomly gauss noise the 2D image.
    """
    def __call__(self, image):
        noise = random.choice((None,1))
        if noise is not None:
            image[image<0] = 0
            level = np.random.uniform(low=0.001, high=0.02, size=1)
            sigma = np.random.uniform(low=0.01, high=0.1, size=1)
            sigma = sigma[0]*image+level[0]
            gauss = torch.normal(0,sigma)
            image = image + gauss
            image[image<0] = 0
        return image
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))
class GaussianBlur(object):
    """Randomly gaussian blur the image
    """

    def __init__(self, kernel_size, sigma_low, sigma_high):
        self.kernel_size = kernel_size
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high

    def __call__(self, image):
        noise = random.choice((None,1))
        if noise is not None:
            #sigma = np.random.uniform(self.sigma_low, self.sigma_high)
            sigma = loguniform(self.sigma_low, self.sigma_high)
            func = torchvision.transforms.GaussianBlur(self.kernel_size, sigma)

            image[image<0] = 0
            if image.ndim == 2:
                image = image.unsqueeze(0)  # add channel dimension
                image = func(image)
                image = image.squeeze(0)  # remove channel dimension
            else:
                image = func(image)
        return image