import os
import glob
import pydicom as pyd
import numpy as np
import pandas as pd
import statistics as stat
import nibabel as nib
from toolbox import progress_bar
debug = 1

# This script is for generating the numpy files utilized for model training
# Performs the calculation of the slope 1 (enhancement) for each scan
# Performs the calculation of the slope 2 (washout) for each scan
# Normalizes samples by dividing by 95th percentile of T1_01_01
if __name__ == '__main__':
    inputDir = '/data/coreg'
    saveDir = '/data/inputs'
    Data_table = pd.read_csv('/data/Data_table_timing.csv')
    session = np.unique(Data_table['SessionID'])

    Dirs = os.listdir(f'{inputDir}/')
    #Dirs.sort()
    N = len(Dirs)
    k = 0

    progress_bar(k,N, status='Generating model inputs...')

    # Check if inputs have already been generated
    if os.path.exists(saveDir):
        print('Inputs already generated')
        print('To reprocess data, please remove /data/inputs')
        #exit()
    else:
        # Create directory for saving inputs
        os.mkdir(saveDir)

    for i in Dirs:
        progress_bar(k,N, status=f'Processing {i.split(os.sep)[-1]}')
        os.mkdir(saveDir + f'/{i.split(os.sep)[-1]}')

        Fils = glob.glob(f'{inputDir}/{i}/*.nii')
        Fils.sort()
        n = len(Fils)
        
        # Extracting timepoints from datatable
        Data = Data_table[Data_table['SessionID'] == i.split(os.sep)[-1]]
        Major = Data['Major'].reset_index(drop=True)
        sorting = np.argsort(Major)
        Times = [Data['SeriesTime'].reset_index(drop=True)[ii] for ii in sorting] #Loading Times in ms
        # Converting to seconds
        Times = [t/1000 for t in Times]

        #Times = []
        #for j in range(0,n):
        #    Times.append(j*15)

        if not n >= 3:# 'Minimum 3 scans required to generate inputs'
            if debug > 0:
                print(f'Not enough scans for {i.split(os.sep)[-1]}')
            continue

        # Load the 01 scan
        img = nib.load(Fils[0])
        data0 = img.get_fdata()
        data0[np.isnan(data0)] = 0
        p95 = float(np.percentile(data0,95))

        # Create a new NIfTI image with the same affine, but with the data type, slope, and intercept set explicitly
        new_img = nib.Nifti1Image(data0, img.affine)
        new_img.header['datatype'] = 16
        new_img.header['scl_slope'] = 1
        new_img.header['bitpix'] = 32
        new_img.header['cal_max'] = 0
        new_img.header['cal_min'] = 0

        # Building time matrix same shape as loaded data
        T = np.zeros_like(data0)
        T = np.expand_dims(T, axis=-1)
        T = np.repeat(T, len(Times), axis=-1)
        for ii,jj in enumerate(Times):
            T[:,:,:,ii] = jj
        
        # Loading all image data into single matrix
        D = np.zeros_like(data0)
        D = np.expand_dims(D, axis=-1)
        D = np.repeat(D, n, axis=-1)
        for ii,jj in enumerate(Fils):
            img = nib.load(jj)
            data0 = img.get_fdata()
            data0[np.isnan(data0)] = 0
            D[:,:,:,ii] = data0
        D[np.isnan(D)] = 0

        ###################################
        # Calculating slope 1 (enhancement)
        Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1)
        Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1)
        print('Slope 1 calculation')
        slope1 = np.divide(
            np.sum((T[:,:,:,0:2] - Tmean) * (D[:,:,:,0:2] - Dmean), axis=3),
            np.sum(np.square((T[:,:,:,0:2] - Tmean)), axis=3)
        ).astype(np.float32)
        slope1 = slope1 / p95

        nib.save(nib.Nifti1Image(slope1.astype('float32'), img.affine), saveDir + f'/{i.split(os.sep)[-1]}/slope1.nii')

        ###################################
        # Calculating slope 2 (washout)
        Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,1:], axis=3), axis=-1), n-1, axis=-1)
        Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,1:], axis=3), axis=-1), n-1, axis=-1)
        print('Slope 2 calculation')
        slope2 = np.divide(
            np.sum((T[:,:,:,1:] - Tmean) * (D[:,:,:,1:] - Dmean), axis=3),
            np.sum(np.square((T[:,:,:,1:] - Tmean)), axis=3)
        ).astype(np.float32)
        slope2 = slope2 / p95

        nib.save(nib.Nifti1Image(slope2.astype('float32'), img.affine), saveDir + f'/{i.split(os.sep)[-1]}/slope2.nii')

        ###################################
        # Creating post-contrast image
        img = nib.load(Fils[1])
        data1 = img.get_fdata()
        data1[np.isnan(data1)] = 0
        post = data1/p95

        nib.save(nib.Nifti1Image(post.astype('float32'), img.affine), saveDir + f'/{i.split(os.sep)[-1]}/post.nii')

        ###################################

        k += 1



