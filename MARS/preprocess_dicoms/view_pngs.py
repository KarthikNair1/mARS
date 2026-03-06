import pandas as pd
import numpy as np
import os
import shutil
#os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
#import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('qtagg') 
#import matplotlib.pyplot as plt
#import cv2

def show_images(df, directory, filename_col='filename'):
    for idx, row in df.iterrows():
        img_path = os.path.join(directory, row[filename_col])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        cv2.imshow('Image Viewer', img)
        key = cv2.waitKey(0)  # Wait indefinitely for a key press

        if key == 27:  # ESC key to break early
            break

    cv2.destroyAllWindows()

# Example: Your DataFrame with filenames (assumed already loaded)
df = pd.read_csv('data_scan_curated.csv') 
# Example column: 'filename'
directory = '/gpfs/scratch/haggsi01/myeloma_lars/process_dicoms_for_lars/images_h5/'

# Only large FOV
df = df[df.axial_fov>1500]

1/0
for i,row in df.iterrows():
    print(row.filename.replace('.h5','.png'))
    shutil.copy('images_thumbnails_png/'+row.filename.replace('.h5','.png'), 'thumbs_large_fov/'+row.filename.replace('.h5','.png'))

#show_images(df,directory,filename_col='filename')