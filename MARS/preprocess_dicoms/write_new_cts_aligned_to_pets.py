import os
import sys
import pandas as pd
import numpy as np
import os
import h5py
import pydicom 
import glob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import scipy

#sys.path.append('../MARS/preprocess_dicoms')
from read_dicom_functions import GetDicomSlices, GetImageFromDicomSlices, GetImageFromDicomSlices_PixelData, ImageBQML2SUV
from utils import show_mips, show_grid, bbox_nd, read_image_h5, read_header_pkl, get_sample_from_df
import pdb
import time
import argparse
from collections import Counter

import SimpleITK as sitk

def get_imaging_plane(orientation):
    plane = np.cross(orientation[0:3], orientation[3:6])
    plane = [abs(x) for x in plane]

    if plane[0] == 1: #Sagittal
        return 'sagittal'
    elif plane[1] == 1: #Coronal
        return 'coronal'
    elif plane[2] == 1: #Transverse
        return 'transverse'
    else:
        print(f'Plane is not recognized, plane = {plane}')
def load_sitk_from_dir(dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

if __name__ == '__main__':
    # argparse to accept start and end index
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_nodes', type=int)
    parser.add_argument('--node_index', type=int)
    #parser.add_argument('--start', type=int, default=0, help='Start index')
    #parser.add_argument('--end', type=int, default=10000, help='End index')
    args = parser.parse_args()
    # Step 1: figure out which ids need to be processed and load in the large df
    filenames = {'CT': {}, 'PT': {}}

    full_pt = pd.read_csv('/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/data_scan_curated.csv')
    assert full_pt.shape[0] == full_pt['filepath'].nunique(), "Duplicate filepaths in full_pt dataframe"
    for idx, row in full_pt.iterrows():
        filenames['PT'][row['filepath']] = row['filename']

    full_ct = pd.read_csv('/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/data_scan_filtered_for_ct_segmentation.csv')
    assert full_ct.shape[0] == full_ct['filepath'].nunique(), "Duplicate filepaths in full_ct dataframe"
    for idx, row in full_ct.iterrows():
        filenames['CT'][row['filepath']] = row['filename']

    # take all ids which have a PT and CT scan as the ones to be processed
    ids_with_pt_and_ct = list(filenames['PT'].keys() & filenames['CT'].keys())
    ids_with_pt_and_ct = sorted(ids_with_pt_and_ct)

    # split using node_index and total_nodes
    start = args.node_index * (len(ids_with_pt_and_ct) // args.total_nodes)
    if args.node_index == args.total_nodes - 1:
        end = len(ids_with_pt_and_ct)
    else:
        end = (args.node_index + 1) * (len(ids_with_pt_and_ct) // args.total_nodes)
    ids_with_pt_and_ct = ids_with_pt_and_ct[start:end]

    print(f'Processing ids from index [{start},{end}) out of {len(ids_with_pt_and_ct)} total ids with both PT and CT scans.')
    
    # load in df containing SUIDs, orientations, etc. which will be helpful in working from the DICOMS
    df = pd.read_csv('/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/data_scan.csv')

    for id in ids_with_pt_and_ct:
        print(f'On {id}')
        output_folder = f'/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_h5_aligned_ct/{id}'
        qc_output_folder = f'/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/qc_ct_alignment_to_pet/{id}'

        # check if exists
        if os.path.exists(os.path.join(qc_output_folder, 'qc_images.png')):
            print(f'QC images already exist for {id}, skipping...')
            continue

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(qc_output_folder, exist_ok=True)

        SLICES = GetDicomSlices(f'/gpfs/data/mayerhoeferlab/projects/myeloma/dicom/{id}')

        L = np.array([np.array(x.ImageOrientationPatient).size for x in SLICES])
        idx = np.where(L==6)[0]
        SLICES = [SLICES[i] for i in idx]

        # Unique series
        SUID = [x.SeriesInstanceUID for x in SLICES]
        SUID = list(set(SUID))

        # get the unique orientations
        ORIENT = np.round(np.array([x.ImageOrientationPatient for x in SLICES]),decimals=2) 
        ORIENT = np.unique(ORIENT,axis=0)

        # sort the unique SUID's and orientations
        SUID = sorted(SUID)
        ORIENT = ORIENT[np.lexsort((ORIENT[:, 5],ORIENT[:, 4],ORIENT[:, 3],ORIENT[:, 2],ORIENT[:, 1],ORIENT[:, 0]))]

        # for every combination of (SUID, orientation), save all slices with this combination
        dicto_slices_for_suid_orient = {}
        for slic in SLICES:
            key = (slic.SeriesInstanceUID, tuple(np.round(slic.ImageOrientationPatient, decimals=2)))
            if key not in dicto_slices_for_suid_orient:
                dicto_slices_for_suid_orient[key] = []
            dicto_slices_for_suid_orient[key].append(slic)

        dir_path_dict = {'CT': '', 'PT': ''}
        for modality in ['CT', 'PT']:
            subdf = df[(df.filepath == id) & (df.modality == modality)]
            row = subdf[subdf.filename == filenames[modality][id]]
            assert(len(row) == 1), f"Expected exactly one row for filepath {id} and modality {modality}, but got {len(row)}"
            orient_str = row['image_orientation'].values[0]
            orient = np.fromstring(orient_str.strip('[]'), sep=',').astype(float)
            orient = np.round(orient, decimals=2)
            key = (row['series_uid'].values[0], tuple(orient))

            slices_here = dicto_slices_for_suid_orient[key]
            sop_uid = np.array([x.SOPInstanceUID for x in slices_here])
            _,idx = np.unique(sop_uid, return_index=True)
            prev = len(slices_here)
            idx = set(idx)
            slices_here = [x for i,x in enumerate(slices_here) if i in idx]

            # write
            base_path = os.path.join('/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/dicoms_rewrite', str(row['filepath'].values[0]), row['filename'].values[0].replace('.h5',''))
            print(base_path)
            dir_path_dict[modality] = base_path
            os.makedirs(base_path, exist_ok=True)
            for i in range(len(slices_here)):
                this_path = os.path.join(base_path, f'slice{i:03d}.dcm')
                slices_here[i].save_as(this_path)

        # now, load as SITK
        img_ct = load_sitk_from_dir(dir_path_dict['CT'])
        img_pt = load_sitk_from_dir(dir_path_dict['PT'])

        # resample CT to PT space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_pt)          # reference geometry
        resampler.SetInterpolator(sitk.sitkLinear)    # linear for continuous data (MRI/CT), nearest for labels
        resampler.SetDefaultPixelValue(-1024)             # value outside original image
        resampler.SetTransform(sitk.Transform())  # identity
        img_resampled = resampler.Execute(img_ct)

        # write resampled image to folder
        arr_resampled = sitk.GetArrayFromImage(img_resampled)
        arr_resampled = arr_resampled.transpose((1,2,0))
        resampled_path = os.path.join(output_folder, filenames['CT'][id])
        with h5py.File(resampled_path, 'w') as f:
            f.create_dataset('image', data=arr_resampled, compression="gzip")

        # plot QC
        fig, axs = plt.subplots(3,3, figsize=(25,25))

        # axis 0: PT loaded from h5
        # axis 1: PT from dicoms
        # axis 2: resampled CT

        # first, get h5
        file = f'/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/images_h5/{id}/{filenames["PT"][id]}'
        file = h5py.File(file, 'r')
        image_data = file['image'][:]
        sz_1 = image_data.shape

        # second, PT from dicoms
        arr_pt = sitk.GetArrayFromImage(img_pt)
        arr_pt = arr_pt.transpose((1,2,0))
        sz_2 = arr_pt.shape

        # third, CT from resampled
        arr_resampled = sitk.GetArrayFromImage(img_resampled)
        arr_resampled = arr_resampled.transpose((1,2,0))
        sz_3 = arr_resampled.shape

        candidate_idxs = [sz_1[2]//10, 5*sz_1[2]//10, 9*sz_1[2]//10]
        for ii, img_idx in enumerate(candidate_idxs):
            axs[ii, 0].imshow(image_data[:,:,img_idx], cmap='gray_r')
            axs[ii, 1].imshow(arr_pt[:,:,img_idx], cmap='gray_r')
            axs[ii, 2].imshow(arr_resampled[:,:,img_idx], cmap='gray')
            for jj in range(3):
                axs[ii,jj].xaxis.set_visible(False)
                axs[ii,jj].yaxis.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(qc_output_folder, 'qc_images.png'))
        print(f'h5 PT shape: {sz_1}, dicom PT shape: {sz_2}, resampled CT shape: {sz_3}')
        print('==========================')


    
