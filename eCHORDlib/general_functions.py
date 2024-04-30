# -*- coding: utf-8 -*-
"""
Fonctions générales
"""
import os
import h5py
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
import cupy as cp
import time
from skimage.morphology import disk, ball
import scipy.ndimage as sci
from skimage.restoration import denoise_nl_means

import bm3d
from pyvsnr import vsnr2d

#______________________________________________________
# h5py functions
def get_dataset_keys(f):
    
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def get_group_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Group) else None)
    return keys

def openH5file():

    '''
    Open a dialog to look for the H5 file to open
    
    Returns (in this order):
        - H5file itself, to be used to get datasets
        - directory of the file
        - file path
        - the list of keys in the H5 file
    '''
   
    filePath, dirFile = getFilePathDialog('theoretical database (*.crddb)')


    # lecture du fichier de profils theoriques
    f = h5py.File(filePath[0], 'r')

    return f, dirFile, filePath[0], get_dataset_keys(f)
    

def savingMTEXfromHDF5():
    # Ouverture du fichier h5
    
    f, dirFile, filePath, listKeys = openH5file()
    
    
    for i in listKeys:
        if "nScoresOri" in i:
            nScoresOri = np.asarray(f[i])
            
    Quat=nScoresOri[-1,:,:,:]
    x = len(Quat[0])
    y = len(Quat[0][0])

    ti = time.strftime("%Y-%m-%d__%Hh-%Mm")

    with open(dirFile + '\indexGPU_'+ ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:
    
        for i in range(x):
            for j in range(y):

                index = 1    
                if Quat[0,i,j] ==0 :
                    index = 0
                file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n') 

#________________________________________________________
# type conversion

def scale_to(x, x_min, x_max, t_min, t_max): # Conversion en 16bits
    """
    Scales x to lie between t_min and t_max
    Links:
         https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
         https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    """
    r = x_max - x_min
    r_t = t_max - t_min
    assert(math.isclose(0,r, abs_tol=np.finfo(float).eps) == False)
    x_s = r_t * (x - x_min) / r + t_min
    return x_s

#________________________________________________________
# GPU function with cupy

def isCupy(a):
    '''
    Parameters
    ----------
    a : any python object
        the object which type to be determined, cupy or not cupy.

    Returns
    -------
    Boolean
        True if the object in parameter is a cupy object, False for any other type.

    '''
    return str(type(a)) == "<class 'cupy.ndarray'>"

#________________________________________________________
# general functions

def getFilePathDialog(text):
    '''
    
    Parameters
    ----------
    text : Ttype 'string'
        the text to be displayed to describe what type of file should be opened by the dialog

    Returns
    -------
    filePath
        the full path to the selected file

    '''
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()

    # Pour utiliser le programme de traitement stack
    filePath = filedialog.askopenfilenames(title=text)
    dirFile = os.path.dirname(filePath[0])
    
    # Pour indexation ces lignes fonctionne (jusque V16 d'indexation sur GPU)
    # filePath = filedialog.askopenfilename(title=text, multiple=True, parent=window)[0]
    # dirFile = os.path.dirname(filePath)
    
    return filePath, dirFile


#______________________________
# image treatment

def remove_outliers(image, radius=2, threshold=50):
    footprint_function = disk if image.ndim == 2 else ball
    footprint = footprint_function(radius=radius)
    median_filtered = sci.median_filter(image, footprint=footprint)
    outliers = (
        (image > median_filtered + threshold)
        | (image < median_filtered - threshold)
    )
    output = np.where(outliers, median_filtered, image)
    return outliers, output

def NonLocalMeanDenoising(image, param_h = 0.06, fastAlgo = True, size = 5, distance = 6):

    denoisedImage = denoise_nl_means(image, h= param_h, fast_mode = True,
                                     patch_size = size,
                                     patch_distance = distance)
    return denoisedImage

def BM3D(image, psd, isFast):
    if isFast:
        denoisedImage = bm3d.bm3d(image, sigma_psd = psd, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    else:
        denoisedImage = bm3d.bm3d(image, sigma_psd = psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    return denoisedImage

def VSNR_funct(image, filter_):
    
    # denoisedImage = vsnr2d(image, filter_, nite=20, xp=np)
    denoisedImage = vsnr2d(image, filter_)
   
    # # Fonctionne avec pyvsnr
    # alpha=filter_[0]['noise_level']
    # name=filter_[0]['name']
    # sigma = filter_[0]['sigma']
    
    # denoisedImage_vsnr = VSNR(image.shape)
    # denoisedImage_vsnr.add_filter(alpha,name,sigma)
    # denoisedImage_vsnr.initialize()
    # denoisedImage = denoisedImage_vsnr.eval(image, maxit=100, cvg_threshold=1e-4)
    
    return denoisedImage

def convertToUint8(stack):
      mn = stack.min()
      mx = stack.max()
    
      mx -= mn
    
      stack = ((stack - mn)/mx) * 255.0
      return np.round(stack).astype(np.uint8)
   
def sliceinfo(image):
    mn = image.min()
    mx = image.max()
    print(f"\n _________________\nle type de donnée est {type(mn)} \n le min est {mn} \n et le max est {mx}")
    
#______________________________
# Denoising characterization

def Pixels_transformation(arr1, arr2): #Permet de connaitre le nombre de pixel ayant été modifiés
    diff = arr2-arr1 #Différences des deux cartes
    
    Unmodified_pixels_number = np.count_nonzero(diff) # Nombre de pixels n'ayant pas été modifiés
    Total_pixels_number = len(diff)*len(diff[0]) # Nombre total de pixels
    Modified_pixels_proportion = Unmodified_pixels_number * 100 / Total_pixels_number
    
    Var = diff
    Modified_pixels_intensity = Var
    del(Var)
    
    Modified_pixels_intensity[Modified_pixels_intensity == 0] = np.nan
    mean_modified_pixels_intensity = np.nanmean(abs(Modified_pixels_intensity), axis=None)
    
    max_intensity_arr = np.max(arr1)
    mean_proportion_intensity = mean_modified_pixels_intensity * 100 / max_intensity_arr
    
    return Modified_pixels_proportion, mean_modified_pixels_intensity, mean_proportion_intensity
    