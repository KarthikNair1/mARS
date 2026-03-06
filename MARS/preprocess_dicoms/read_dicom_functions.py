import numpy as np 
import pydicom 
from pydicom import dcmread 
from pydicom.pixel_data_handlers.util import apply_modality_lut # rescale and slope
import os
import glob 
import pdb
import pandas as pd

# get all dicom slices from a scan folder
#---------------------------------------------------------
def GetDicomSlices(scan_path):
    #pdb.set_trace()
	# create empty matrix
    files_paths = []
    files = []
    ordered_files = []
    # get file names
    #file_names = os.listdir(scan_path)
    file_names = [os.path.basename(x) for x in glob.glob(scan_path+'/*.dcm')]
    # check that there are slices
    if file_names:
        # get path to all .dcm files in scan folder
        for f in file_names:
            #if f.endswith('.dcm'):
            files_paths.append(os.path.join(scan_path,f))
        
        if files_paths == []:
            print("No slices in scan {}".format(scan_path))
            ordered_files = []
        #pdb.set_trace()
        # read files and create 3D slices matrix
        for file_path in files_paths:
            # check file_path is correct
            if os.path.isfile(file_path):
				# read file
                f = dcmread(file_path)
                # f = dcmread(file_path,force=True)
				# add to slice matrix
                files.append(f)
        
        #pdb.set_trace()
        # save slices with location information
        for f in files: 
            #if hasattr(f,'SliceLocation'): 
            if hasattr(f,'ImagePositionPatient'): 
                #if f.SliceLocation is not None:
                if f.ImagePositionPatient is not None:
                    ordered_files.append(f)
        
        #pdb.set_trace()
     	# order slices 
        try:
            #ordered_files = sorted(ordered_files, key=lambda s: s.SliceLocation)
            ordered_files = sorted(ordered_files, key=lambda s: s.ImagePositionPatient[2], reverse=False)
        except:
            print("Unknown slice location in scan {}".format(scan_path))
            ordered_files = []
    else: 
        print("No slices in scan {}".format(scan_path))
        ordered_files = []
    
    return ordered_files

#---------------------------------------------------------
def GetDicomSlices_metaOnly(scan_path):
    #pdb.set_trace()
	# create empty matrix
    files_paths = []
    files = []
    ordered_files = []
    # get file names
    #file_names = os.listdir(scan_path)
    file_names = [os.path.basename(x) for x in glob.glob(scan_path+'/*.dcm')]
    # check that there are slices
    if file_names:
        # get path to all .dcm files in scan folder
        for f in file_names:
            #if f.endswith('.dcm'):
            files_paths.append(os.path.join(scan_path,f))
        
        if files_paths == []:
            print("No slices in scan {}".format(scan_path))
            ordered_files = []
        #pdb.set_trace()
        # read files and create 3D slices matrix
        for file_path in files_paths:
            # check file_path is correct
            if os.path.isfile(file_path):
				# read file
                f = dcmread(file_path, stop_before_pixels=True)
                # f = dcmread(file_path,force=True)
				# add to slice matrix
                files.append(f)
        
        #pdb.set_trace()
        # save slices with location information
        for f in files: 
            #if hasattr(f,'SliceLocation'): 
            if hasattr(f,'ImagePositionPatient'): 
                #if f.SliceLocation is not None:
                if f.ImagePositionPatient is not None:
                    ordered_files.append(f)
        
        #pdb.set_trace()
     	# order slices 
        try:
            #ordered_files = sorted(ordered_files, key=lambda s: s.SliceLocation)
            ordered_files = sorted(ordered_files, key=lambda s: s.ImagePositionPatient[2], reverse=False)
        except:
            print("Unknown slice location in scan {}".format(scan_path))
            ordered_files = []
    else: 
        print("No slices in scan {}".format(scan_path))
        ordered_files = []
    
    return ordered_files
   
# extract slice pixels and save in 3D np.array of the full scan
#---------------------------------------------------------
def GetImageFromDicomSlices(slice_files):  
    # get scan info
    nr_of_slices = len(slice_files) # nr of slices
    slice_size = np.shape(slice_files[0].pixel_array) # size of each slice
    # create zero matrix
    img3D_shape = [slice_size[0], slice_size[1], nr_of_slices]
    img3D = np.zeros(img3D_shape)
    # extract pixel values of each slice and save as a 3D image
    for i, s in enumerate(slice_files): 
        arr = s.pixel_array
        img3D[:,:,i] = arr 
        
        # transform to corrrect units 
        img3D[:,:,i] = apply_modality_lut(img3D[:,:,i], s)
	      
    return img3D
# extract slice pixels and save in 3D np.array of the full scan
#---------------------------------------------------------
def GetImageFromDicomSlices_PixelData(slice_files):  
    # get scan info
    nr_of_slices = len(slice_files) # nr of slices
    slice_size = np.shape(slice_files[0].PixelData) # size of each slice
    # create zero matrix
    img3D_shape = [slice_size[0], slice_size[1], nr_of_slices]
    img3D = np.zeros(img3D_shape)
    # extract pixel values of each slice and save as a 3D image
    for i, s in enumerate(slice_files): 
        arr = s.PixelData
        img3D[:,:,i] = arr 
        
        # transform to corrrect units 
        img3D[:,:,i] = apply_modality_lut(img3D[:,:,i], s)
	      
    return img3D
# extract slice pixels and save in 3D np.array of the full scan
#---------------------------------------------------------
def ImageBQML2SUV(img,df): 
    ## decay correction to aquisition start
    #pdb.set_trace()
    if not ('scan_time' in df and 'tracer_injection_time' in df and  'patient_weight' in df and 'tracer_halflife' in df):
        return 0,np.nan
    elif not (~df.scan_time.isna().values[0] and ~df.tracer_injection_time.isna().values[0] and ~df.patient_weight.isna().values[0] and ~df.tracer_halflife.isna().values[0]):
        return 0,np.nan
    
    #pdb.set_trace()
    # Calculate the decay: decayFactor = e^(t1-t2/halflife);
    try:
        scan_time = pd.to_datetime(df.scan_time, format='%H%M%S.%f')
    except:
        scan_time = pd.to_datetime(df.scan_time, format='%H%M%S')
    try:
        inj_time = pd.to_datetime(df.tracer_injection_time, format='%H%M%S.%f')
    except:
        inj_time = pd.to_datetime(df.tracer_injection_time, format='%H%M%S')
    #pdb.set_trace()
    dt = (scan_time - inj_time).dt.total_seconds() # [sec]
    decay = np.exp(-np.log(2)*dt/df.tracer_halflife.values[0])
    # Calculate the dose decayed to scan start
    injected_dose_atscan = df.tracer_dose*decay; # in Bq
    #print('decay factor:',decay)
    #print('inj dose@scan:',injected_dose_atscan/1e6,'MBq')
    #pdb.set_trace()
    # Calculate BQML -> SUV. SUV = img[Bq/ml] * Patient Weight[g] / Dose after decay[Bq]
    factor = ((df.patient_weight*1e3)/injected_dose_atscan).values[0]
    img = img*factor
    
    # set all background pixels to zero
    img[img<0] = 0
    
    return img,factor
 
# Saving as binary int16
#---------------------------------------------------------
def SaveScanAsBinary(img3D, filename, directory):
    # # get img shape 
    # size = np.shape(img3D)
    # make dir if non existent 
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # save file 
    img3D.astype('float32').tofile( os.path.join(directory,filename) )
