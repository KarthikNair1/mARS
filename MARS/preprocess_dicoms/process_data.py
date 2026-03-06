import os
import sys
#sys.path.append('../../MARS/preprocess_dicoms')
#sys.path.append('../../MARS/run_lars')

import pandas as pd
import numpy as np
import os
import h5py
import pydicom 
import glob
#import tkinter
#os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#import itertools
import pickle
import scipy
from read_dicom_functions import GetDicomSlices, GetImageFromDicomSlices, GetImageFromDicomSlices_PixelData, ImageBQML2SUV
from utils import show_mips, show_grid, bbox_nd, read_image_h5, read_header_pkl, get_sample_from_df
import pdb
import time
import argparse
from collections import Counter

import warnings
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
)

#############
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

def create_info_df_from_slice(slices, study_id_number, filename, pix_size_3, slice_diff, imaging_plane):
    tmp = pd.DataFrame()
    tmp.loc[0,'filepath'] = study_id_number
    tmp.loc[0,'filename'] = filename+'.h5'
    tmp.loc[0,'filename_header'] = filename+'.pkl'
    tmp.loc[0,'fullfile'] = study_id_number +'/'+ tmp.loc[0,'filename'] 
    tmp.loc[0,'patient_id'] = slices[0].PatientID
    #if 'PatientBirthDate' in slices[0]:
    #    tmp.loc[0,'birth_date'] = slices[0].PatientBirthDate
    if 'PatientAge' in slices[0]: 
        tmp.loc[0,'patient_age'] = slices[0].PatientAge
    if 'PatientSex' in slices[0]: 
        tmp.loc[0,'patient_sex'] = slices[0].PatientSex
    if 'PatientWeight' in slices[0]: 
        tmp.loc[0,'patient_weight'] = slices[0].PatientWeight
    if 'SmokingStatus' in slices[0]: 
        tmp.loc[0,'smoking_status'] = slices[0].SmokingStatus
    tmp.loc[0,'accession'] = slices[0].AccessionNumber
    tmp.loc[0,'study_uid'] = slices[0].StudyInstanceUID
    tmp.loc[0,'series_uid'] = slices[0].SeriesInstanceUID
    tmp.loc[0,'scan_date'] = str(slices[0].StudyDate)
    if 'SeriesTime' in slices[0]: 
        tmp.loc[0,'scan_time'] = str(slices[0].SeriesTime)
    tmp.loc[0,'modality'] = slices[0].Modality
    tmp.loc[0,'matrix_size_1'] = int(slices[0].Columns)
    tmp.loc[0,'matrix_size_2'] = int(slices[0].Rows)
    tmp.loc[0,'matrix_size_3'] = int(len(slices))
    if type(slices[0].PixelSpacing)==pydicom.multival.MultiValue:
        tmp.loc[0,'pixel_size_1'] = slices[0].PixelSpacing[0]
        tmp.loc[0,'pixel_size_2'] = slices[0].PixelSpacing[1]
    elif type(slices[0].PixelSpacing)==pydicom.valuerep.DSfloat:
        tmp.loc[0,'pixel_size_1'] = float(slices[0].PixelSpacing)
        tmp.loc[0,'pixel_size_2'] = float(slices[0].PixelSpacing)
    tmp.loc[0,'pixel_size_3'] = pix_size_3
    if len(slice_diff)>0:
        counts = Counter(slice_diff)
        sliceloc = str(dict(counts))
    else:
        sliceloc = '{'+str(pix_size_3)+': 1}'
    tmp.loc[0,'pixel_size_3_str'] = sliceloc
    if 'SliceThickness' in slices[0]: 
        tmp.loc[0,'slice_thickness'] = slices[0].SliceThickness
    if 'SeriesDescription' in slices[0]: 
        tmp.loc[0,'description_series'] = slices[0].SeriesDescription
    if 'StudyDescription' in slices[0]: 
        tmp.loc[0,'description_study'] = slices[0].StudyDescription
    if 'AdmittingDiagnosesDescription' in slices[0]: 
        tmp.loc[0,'admitting_diagnoses'] = slices[0].AdmittingDiagnosesDescription
    if 'ProtocolName' in slices[0]: 
        tmp.loc[0,'protocol_name'] = slices[0].ProtocolName
    if 'ImageComments' in slices[0]: 
        tmp.loc[0,'image_comments'] = slices[0].ImageComments 
    if 'BodyPartExamined' in slices[0]: 
        tmp.loc[0,'body_part'] = slices[0].BodyPartExamined
    if 'Manufacturer' in slices[0]: 
        tmp.loc[0,'manufacturer'] = slices[0].Manufacturer
    if 'ManufacturerModelName' in slices[0]: 
        tmp.loc[0,'manufacturer_model'] = slices[0].ManufacturerModelName
    if 'SoftwareVersions' in slices[0]: 
        tmp.loc[0,'software'] = str(slices[0].SoftwareVersions)
    if 'ReconstructionMethod' in slices[0]: 
        tmp.loc[0,'reconstruction'] = slices[0].ReconstructionMethod
    if 'ConvolutionKernel' in slices[0]: #[0x00181210]
        tmp.loc[0,'convolution_kernel'] = str(slices[0].ConvolutionKernel)
    if (0x00431064) in slices[0]: #[0x00431064]
        tmp.loc[0,'image_filter'] = slices[0][0x00431064]
    if 'CorrectedImage' in slices[0]: 
        tmp.loc[0,'corrections'] = '|'.join(slices[0].CorrectedImage)
    if 'Units' in slices[0]: 
        tmp.loc[0,'pixel_unit'] = slices[0].Units
    tmp.loc[0,'factor_bqml2suv'] = np.nan
    if 'ActualFrameDuration' in slices[0]: 
        tmp.loc[0,'frame_duration'] = slices[0].ActualFrameDuration
    if 'RadiopharmaceuticalInformationSequence' in slices[0]: 
        if 'Radiopharmaceutical' in slices[0].RadiopharmaceuticalInformationSequence[0]: 
            tmp.loc[0,'tracer'] = slices[0].RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical
        if 'RadiopharmaceuticalStartTime' in slices[0].RadiopharmaceuticalInformationSequence[0]: 
            tmp.loc[0,'tracer_injection_time'] = str(slices[0].RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
        if 'RadionuclideTotalDose' in slices[0].RadiopharmaceuticalInformationSequence[0]: 
            tmp.loc[0,'tracer_dose'] = slices[0].RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose 
        if 'RadionuclideHalfLife' in slices[0].RadiopharmaceuticalInformationSequence[0]: 
            tmp.loc[0,'tracer_halflife'] = slices[0].RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    if 'XRayTubeCurrent' in slices[0]: 
        tmp.loc[0,'xray_tube_current'] = slices[0].XRayTubeCurrent
    if 'KVP' in slices[0]: 
        tmp.loc[0,'kvp'] = slices[0].KVP
    if 'ContrastBolusAgent' in slices[0]: 
        tmp.loc[0,'contrast_agent'] = slices[0].ContrastBolusAgent
    if 'ContrastBolusRoute' in slices[0]: 
        tmp.loc[0,'contrast_route'] = slices[0].ContrastBolusRoute
    if 'KVP' in slices[0]: 
        tmp.loc[0,'kvp'] = slices[0].KVP
    if 'PatientPosition' in slices[0]: 
        tmp.loc[0,'patient_position'] = slices[0].PatientPosition
    tmp.loc[0,'imaging_plane'] = imaging_plane
    if 'ImageOrientationPatient' in slices[0]: 
        tmp.loc[0,'image_orientation'] = str(slices[0].ImageOrientationPatient)
    if 'AdditionalPatientHistory' in slices[0]: 
        tmp.loc[0,'patient_history'] = slices[0].AdditionalPatientHistory
    
    return tmp

def fix_patient_position(tmp, img):
    # Rotate to FFS
    if 'patient_position' not in tmp:
        tmp.loc[0,'patient_position'] = 'HFS'
        print('          EMPTY-->HFS') 
    elif (pd.isnull(tmp['patient_position'].values[0])): 
        tmp.loc[0,'patient_position'] = 'HFS'
        print('          EMPTY-->HFS')                
    elif (tmp['patient_position'].values[0]==''): # manually checked single case
        tmp.loc[0,'patient_position'] = 'HFS'
        print('          EMPTY-->HFS')

    if tmp.patient_position.values[0]=='FFDR':
        img = np.flip(img.transpose(1,0,2),axis=1)
        ps1 = tmp.pixel_size_2.values[0]
        ps2 = tmp.pixel_size_1.values[0]
        tmp.loc[0,'pixel_size_1'] = ps1
        tmp.loc[0,'pixel_size_2'] = ps2
        print('          FFDR-->FFS')
    
    return tmp, img
#############


parser = argparse.ArgumentParser(description='Process DICOMs')
parser.add_argument('-N', default=1, type=int, help='Number of splits of the dataset')
parser.add_argument('-n', default=0, type=int, help='Which of N datasplits to process')
parser.add_argument('-n2', default=0, type=int, help='Image index in datasplit to start from')
parser.add_argument('-n3', default=-1, type=int, help='Image index in datasplit to end with')
parser.add_argument('--dataset_dir', default='', type=str, help='Directory containing dicom directories, i.e. dataset_dir/* gives all the dicom folders.')
parser.add_argument('--save_dir', default='', type=str, help='Directory to save processed images, headers, thumbnails, and datafiles to.')

plt.rcParams.update({'figure.max_open_warning': 0})

def split_dirs_last_chunk_remainder(dirs, N):
    M = len(dirs)
    base_size = M // N
    remainder = M % N

    chunks = []
    for i in range(N):
        start = i * base_size
        end = (i + 1) * base_size
        chunks.append(dirs[start:end])

    # Add the remainder to the last chunk
    if remainder > 0:
        chunks[-1].extend(dirs[N * base_size:])

    return chunks

args = parser.parse_args()
print(args)

### Directory containing dicom directories
main_dir = args.dataset_dir
rt_save_dir = args.save_dir
os.makedirs(rt_save_dir, exist_ok=True)
savedir_i = os.path.join(rt_save_dir, 'images_h5') #where to save images 
savedir_h = os.path.join(rt_save_dir, 'headers_pkl') #where to save headers
savedir_csv = os.path.join(rt_save_dir, 'datafiles') #where to save csvs
savedir_thumbnails = os.path.join(rt_save_dir, 'image_thumbnails_png') # where to save image thumbnails

DIRS = [ f.path for f in os.scandir(main_dir) if f.is_dir() ]
ti = 0
t0 = time.time()
avg_time_per_study = 0

# Split DIRS
chunks = split_dirs_last_chunk_remainder(DIRS, args.N)
DIRS_sub = chunks[args.n]
if args.n3==-1:
    DIRS_sub = DIRS_sub[args.n2:]
else:
    DIRS_sub = DIRS_sub[args.n2:args.n3]

#1/0
# Allocate
#appended_data = [None]*len(DIRS)
appended_data = []
skip_data = []

# csvfile = 'data_scan_'+str(args.n)+'.csv'
# skipfile = 'data_skipped_'+str(args.n)+'.csv'
csvfile = 'data_scan_'+str(args.n)+'_'+str(args.n2)+'.csv'
skipfile = 'data_skipped_'+str(args.n)+'_'+str(args.n2)+'.csv'

#1/0
# Loop over study directories
print('')
ch = np.empty(0)
for ii,study in enumerate(DIRS_sub):
    t = time.time()
    print('[{:04d}/{:04d}] {}'.format(ii,len(DIRS_sub)-1,study))
    STUDY_ID_NAME = os.path.relpath(study, main_dir).split(os.sep)[0]

    file_counter = len(glob.glob(study+'/*.dcm'))
    SLICES = GetDicomSlices(study)
    if SLICES==[]: continue

    # Remove strange image positions, orientations
    if 'ImageOrientationPatient' not in SLICES[0]:
        continue

    if SLICES[0].ImageOrientationPatient is None:
            continue
    
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

    mm_pt = -1
    mm_ct = -1

    # For each combination of SUID and ORIENT, process the slices:
    for kk,suid in enumerate(SUID):
        for ll,orient in enumerate(ORIENT):
            slices_here = dicto_slices_for_suid_orient.get((suid, tuple(orient)), [])
            if slices_here == []:
                continue
            sop_uid = np.array([x.SOPInstanceUID for x in slices_here])
            _,idx = np.unique(sop_uid, return_index=True)
            prev = len(slices_here)
            idx = set(idx)
            slices_here = [x for i,x in enumerate(slices_here) if i in idx]
            

            # Check coronal, sagittal, transverse plane and sort slices accordingly
            ipp = np.array([x.ImagePositionPatient for x in slices_here])
            imaging_plane = get_imaging_plane(orient)
            if imaging_plane is None:
                print('        skipping   {} {}  (imaging plane not recognized)'.format(study.split('/')[-1].replace(' ','_'), suid))
                continue
            coord_index = {'sagittal': 0, 'coronal': 1, 'transverse': 2}[imaging_plane]

            # sort on the coord index
            idx = np.argsort(ipp[:, coord_index])

            #if imaging_plane == 'transverse':
                # make sure that idx = [0, 1, 2, ...] as it should already be sorted
                # actually, it turns out that this is not necessarily true for all the scans
                #assert np.all(idx == np.arange(len(slices_here))), "Transverse slices not sorted correctly by default"

            # reorder slices and ipp according to our new sorted order
            slices_here = [slices_here[i] for i in idx]
            ipp = np.array([ipp[i,:] for i in idx])

            # Slice thickness in this orientation
            slice_diff = np.round(np.abs(np.diff(ipp[:,coord_index])),decimals=4)
            if len(slice_diff)>0: # i.e. if there are at least 2 slices
                pix_size_3 = scipy.stats.mode(slice_diff)[0]
            else:
                pix_size_3 = ipp[0,coord_index] # just use the value of the coordinate if only one slice
            
            # Add info to df
            study_id_number = study.split('/')[-1].replace(' ','_')
            filename = study_id_number + '_' + slices_here[0].Modality.lower()

            tmp = create_info_df_from_slice(slices_here, study_id_number, filename, pix_size_3, slice_diff, imaging_plane)
            # try to get the image
            try:
                img = GetImageFromDicomSlices(slices_here)
            except:
                try:
                    img = GetImageFromDicomSlices_PixelData(slices_here)
                except:
                    #do nothing
                    print('      skipping   {}'.format(tmp.filename.values[0]))
                    tmp = tmp.astype({col: "float64" for col in tmp.columns if tmp[col].isna().all()}) # if all entries are NaN in a column, make sure its float so we avoid a pandas deprecation warning
                    skip_data.append(tmp)
                    dfskip = pd.concat(skip_data, ignore_index=True)
                    
                    # Write dataframe to disk
                    dfskip.to_csv(os.path.join(savedir_csv,skipfile),index=0)
                    continue
            if tmp.modality.values[0] == 'PT':
                if img.ndim < 3:
                    print('        cannot convert   {}   to SUV'.format(tmp.filename.values[0]))
                    continue
                img,factor = ImageBQML2SUV(img,tmp)
                tmp.loc[0,'factor_bqml2suv'] = factor
            img = np.array(img)
            # Skip scout
            if ('scout' in tmp.description_series.values[0].lower()) | ('mip' in tmp.description_series.values[0].lower()) | (img.ndim<3):
                print('        skipping   {} (SCOUT or MIP)'.format(tmp.filename.values[0]))
                continue
            elif (img.shape[2]<2):
                print('        skipping   {} (SCOUT or MIP)'.format(tmp.filename.values[0]))
                continue

            # fix patient position field to HFS / FFDR
            tmp, img = fix_patient_position(tmp, img)

            ## Filename
            filename = study.split('/')[-1].replace(' ','_')+'_'+tmp.modality.values[0].lower()
            mm = 0
            if tmp.modality.values[0]  == 'PT':
                mm_pt+=1
                mm = mm_pt
            elif tmp.modality.values[0] == 'CT':
                mm_ct+=1
                mm = mm_ct
            if (len(SUID) > 1 or ORIENT.shape[0] > 1):
                filename = filename+'_'+str(mm)
            tmp.loc[0,'filename'] = filename+'.h5'
            tmp.loc[0,'filename_header'] = filename+'.pkl'

            tmp.loc[0,'fullfile'] = STUDY_ID_NAME +'/'+ tmp.loc[0,'filename'] 

            if imaging_plane=='transverse':
                print('        {}\t{}'.format(tmp.loc[0,'filename'],img.shape))
            else:
                print('        {}\t{} ({})'.format(tmp.loc[0,'filename'],img.shape,imaging_plane[0:3].upper()))
            
            
            tmp.loc[0,'img_max'] = img.max()
            tmp.loc[0,'img_min'] = img.min()
            tmp.loc[0,'img_mean'] = img.mean()
            tmp.loc[0,'img_std'] = img.std()
            tmp.loc[0,'img_sum'] = img.sum()

            img_nonzero = np.empty(0)
            if tmp.modality.values[0] == 'CT':
                img_nonzero = img[img>-950]
            if tmp.modality.values[0] == 'PT':
                img_nonzero = img[img>0.1]
            if len(img_nonzero)!=0:
                tmp.loc[0,'img_count_nonzeros'] = len(img_nonzero)
                tmp.loc[0,'img_mean_nonzeros'] = img_nonzero.mean()
                tmp.loc[0,'img_std_nonzeros'] = img_nonzero.std()
                tmp.loc[0,'img_sum_nonzeros'] = img_nonzero.sum()

            appended_data.append(tmp)

            # Plot thumbnails
            if (tmp.modality.values[0] == 'PT') & (img.ndim > 2):
                if ('ATTN' in tmp['corrections'].values[0]) & (tmp['pixel_unit'].values[0]=='BQML') & (tmp['imaging_plane'].values[0]=='transverse'):
                    if tmp.pixel_size_3.values[0]!=0:
                        ps = [tmp.pixel_size_1.values[0],tmp.pixel_size_2.values[0],tmp.pixel_size_3.values[0]]
                    else:
                        ps = [tmp.pixel_size_1.values[0],tmp.pixel_size_2.values[0],tmp.slice_thickness.values[0]]
                    if ps[2]==0: ps[2]=1
                    show_mips(img,figno=mm,pixelsize=ps,savename=os.path.join(savedir_thumbnails, STUDY_ID_NAME, tmp['patient_position'].values[0], filename))

            # Save image
            img_save_path = os.path.join(savedir_i, STUDY_ID_NAME)
            os.makedirs(img_save_path, exist_ok=True)
            img_save_path = os.path.join(img_save_path, str(tmp.filename.values[0]))
            hf = h5py.File(img_save_path, 'w')
            hf.create_dataset('image', data=img, compression='gzip')
            hf.close()
            #print(f'Wrote Image to {img_save_path}')

            # Save header
            for b in range(len(slices_here)):
                if 'PixelData' in slices_here[b]:
                    del slices_here[b].PixelData
                #slices[b].pixel_array = None
            header_save_path = os.path.join(savedir_h, STUDY_ID_NAME)
            os.makedirs(header_save_path , exist_ok=True)
            header_save_path = os.path.join(header_save_path, str(tmp.loc[0,'filename_header']))
            with open(header_save_path, "wb") as hf:
                pickle.dump(slices_here, hf)
                
    # Write dataframe to disk every study loop
    if appended_data != []:
        df = pd.concat(appended_data)
        #df = pd.concat([d for d in appended_data if not d.empty])
        df.to_csv(os.path.join(savedir_csv,csvfile),index=0)
        
    now = time.time()
    ti = ti + 1
    #avg_time_per_study = (now-t)
    print('TIME - tot {:.0f} min | {:.0f} sec ({:.0f} sec/study) | ~{:.0f} min left'.format((now-t0)/60,(now-t),(now-t0)/ti,(now-t0)/ti*(len(DIRS_sub)-ii-1)/60))

# Write dataframe to disk at the end
df = pd.concat(appended_data)
df.to_csv(os.path.join(savedir_csv,csvfile),index=0)
print('Done.')