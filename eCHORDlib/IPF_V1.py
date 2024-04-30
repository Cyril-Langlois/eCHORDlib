# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:37:15 2023

@author: glhote1
"""

import numpy as np

import matplotlib.pyplot as plt
import diffpy.structure as dfs

import Dans_Diffraction as da
import LibrairiesCyril.Xallo as xa
from LibrairiesCyril import Symetry as sy

from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation, symmetry
from orix import plot as orixPlot
from orix.vector import Vector3d

import tkinter as tk
from tkinter import filedialog

from LibrairiesCyril import general_functions as gf

root = tk.Tk()       # initialisation du dialogue
root.withdraw()

#%% réarrangement du tableau d'indexation
# pour le meilleur score déjà

def IPF_Z(quats, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.zvector()):
    
    array2 = np.rot90(quats, 1, (1, 3))
    QuatCorr = np.rot90(array2, 3, (1, 2))
    # maintenant on a la largeur selon l'axe 3, la hauteur selon l'axe 2, et les quaternions selon l'axe 4 et le score suivant axe 0
        
    width = len(QuatCorr[0, 0, :, 0]) # largeur selon l'axe 1
    height = len(QuatCorr[0, :, 0, 0]) # hauteur selon l'axe 0
     
    # importation de la stack au format orix
    
    page = np.zeros((nScores,height * width, 7))
    k = 0
    for a in range(nScores):
        for i in range(height):
            for j in range(width):
                page[a,k, 0] = 1
                page[a,k, 1] = i
                page[a,k, 2] = j
                page[a,k, 3] = QuatCorr[a,i, j, 0]
                page[a,k, 4] = QuatCorr[a,i, j, 1]
                page[a,k, 5] = QuatCorr[a,i, j, 2]
                page[a,k, 6] = QuatCorr[a,i, j, 3]
        
                k += 1
        k = 0
 
    # ré-importation
    phase_id = page[:,:, 0]
    y = page[:,:, 1]
    x = page[:,:, 2]
    quats = page[:,:, 3:]
    
    # conversion Quaternion -> axe-angle car Orix ne permet pas l'import de quaternion expérimentaux !
    axes = np.zeros((nScores,len(quats[0]), 3))
    angles = np.zeros((nScores,len(quats[0]),1))
    
    for score in range(nScores):
        for i in range(len(quats[0])):
            a, b = xa.QuaternionToAxisAngle(quats[score,i, :])
            axes[score,i, :] = a
            angles[score,i] = b
    
    rotations = []
    
    for i in range(nScores):
        axes_i = axes[i,:,:]
        angles_i = angles[i,:,0]
        rotations_i = Rotation.from_axes_angles(axes_i, angles_i, degrees= True)
        rotations.append(rotations_i)
    
    phase_list = PhaseList(
        names=[PhaseName],
        point_groups=[PG],
        structures=phases[0])
    
    # Create a CrystalMap instance
    xmap2 = []
    
    for i in range(nScores):
        xmap2_i = CrystalMap(rotations=rotations[i],phase_id=phase_id[0,:],x=x,y=y,phase_list=phase_list)
        xmap2_i.scan_unit = "um"
        xmap2.append(xmap2_i)

    # information pour utiliser le code couleur spécifique au groupe ponctuel
    pg_laue = xmap2[0].phases[1].point_group.laue
    ipf_key = orixPlot.IPFColorKeyTSL(pg_laue)
    
    o_Cu = []
    for i in range(nScores):
        Var_o_Cu = xmap2[i][PhaseName].orientations
        o_Cu.append(Var_o_Cu)
    
    # Orientation colors
    Ipf_dir = Ipf_dir
    
    ipf_key = orixPlot.IPFColorKeyTSL(pg_laue, direction=Ipf_dir)
    # ipf_key = orixPlot.IPFColorKeyTSL(pg_laue, direction=Vector3d.zvector())
    
    rgb=[]
    for i in range(nScores):
        rgb_i = ipf_key.orientation2color(o_Cu[i])
        rgb.append(rgb_i)
    # rgb est l'image IPF sous forme d'un tableau numpy correctement orienté
    # rgb est flatten, il faut reshaper

    rgb = np.dstack(rgb)
    rgb = np.reshape(rgb,(height, width, 3,nScores))

    rgb = np.swapaxes(rgb, 0, 3)

    rgb = np.swapaxes(rgb, 2, 3)
    rgb = np.flip(rgb, 0)

    # Affichage map 

    rgb = rgb[0,:,:,:]
    rgb = np.flip(rgb,1)
    rgb = np.rot90(rgb)
    
    return rgb, xmap2

def Display_IPF(CIFpath, quats, nScores):
    
    nScores = nScores
    phases = []
    # try:
        # tentative pour utiliser le même fichier CIF utilisé pour l'indexation
        # s'il est toujours au même endroit !
    phases.append(dfs.loadStructure(CIFpath))
    crys = da.functions_crystallography.readcif(CIFpath)
    SymQ = sy.get_proper_quaternions_from_CIF(CIFpath)
        
    # except:
    #     print("Fichier CIF non trouvé !")
    #     # sinon, c'est à l'utilisateur de le localiser
    #     fileList = filedialog.askopenfilename(title='fichier CIF', multiple=True)[0]
    #     phases.append(diffpy.structure.loadStructure(fileList))
    #     crys = da.functions_crystallography.readcif(fileList)
    #     SymQ = sy.get_proper_quaternions_from_CIF(fileList)

    PhaseName = crys["_chemical_formula_sum"]
    numSG = crys["_space_group_IT_number"]
    print(PhaseName)
    PG = symmetry.get_point_group(int(numSG), True).name
    print(PG)
    array = quats[-1, :, :, :] # de base, les quaternions sont sur l'axe 1, on les veut sur l'axe 3

    height = len(array[0])
    width = len(array[0][0])
    
    IPFim_X, xmap = IPF_Z(quats, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.xvector())
    IPFim_Y, xmap = IPF_Z(quats, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.yvector())
    IPFim_Z, xmap = IPF_Z(quats, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.zvector())

    # subplots

    f, axarr = plt.subplots(1,3) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    arr1 = axarr[0].imshow(IPFim_X, interpolation='none')
    arr2 = axarr[1].imshow(IPFim_Y, interpolation='none')
    arr3 = axarr[2].imshow(IPFim_Z, interpolation='none')

    arr1.axes.get_xaxis().set_visible(False)
    arr1.axes.get_yaxis().set_visible(False)

    arr2.axes.get_xaxis().set_visible(False)
    arr2.axes.get_yaxis().set_visible(False)

    arr3.axes.get_xaxis().set_visible(False)
    arr3.axes.get_yaxis().set_visible(False)

    axarr[0].set_title('IPF-X',fontsize = 50)
    axarr[1].set_title('IPF-Y',fontsize = 50)
    axarr[2].set_title('IPF-Z',fontsize = 50)

    plt.show()
    
    return PhaseName, PG, phases, array

#%%

if __name__ == '__main__':
    
    # relecture des données

    # def get_dataset_keys(f):
    #     keys = []
    #     f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    #     return keys

    # def get_group_keys(f):
    #     keys = []
    #     f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Group) else None)
    #     return keys

    # Ouverture du fichier h5

    f, dirFile, fileList, listKeys = gf.openH5file()

    # window = Tk()
    # window.wm_attributes('-topmost', 1)
    # window.withdraw()

    # fileList = filedialog.askopenfilename(title='fichier indexation GPU (*.hdf5)', multiple=True, parent=window)[0]
    # dirFile = os.path.dirname(fileList)

    # # lecture du fichier de profils theoriques test
    # f = h5py.File(fileList, 'r')

    # listKeys = gf.get_dataset_keys(f)
    # # listKeys = get_dataset_keys(f)

    for i in listKeys:
        if "nScoresOri" in i:
            nScoresOri = np.asarray(f[i])

    # Infos sur le cristal 
    # ne fonctionne pour l'instant que pour un monophasé
    # mais déjà une partie sous forme de liste pour gérer plus tard le multiphasé
    CIFpath = []
    nbPhases = 1
    phases = [] 
    nScores = len(nScoresOri[:, 0, 0, 0])

    for h in range(nbPhases):
        # préparation du code pour le cas multiphasé...
        # ouverture des fichiers CIF
        # listKeys = get_group_keys(f)
        listKeys = gf.get_group_keys(f)

        for i in listKeys:
            if "indexation" in i:
                for k in f[i].attrs.keys():
                    if "CIF path" in k:
                        CIFpath.append(f[i].attrs[k])

    try:
        # tentative pour utiliser le même fichier CIF utilisé pour l'indexation
        # s'il est toujours au même endroit !
        phases.append(dfs.loadStructure(CIFpath[0]))
        crys = da.functions_crystallography.readcif(CIFpath[0])
        SymQ = sy.get_proper_quaternions_from_CIF(CIFpath[0])
        
    except:
        print("Fichier CIF non trouvé !")
        # sinon, c'est à l'utilisateur de le localiser
        fileList = filedialog.askopenfilename(title='fichier CIF', multiple=True)[0]
        phases.append(dfs.loadStructure(fileList))
        crys = da.functions_crystallography.readcif(fileList)
        SymQ = sy.get_proper_quaternions_from_CIF(fileList)

    PhaseName = crys["_chemical_formula_sum"]
    numSG = crys["_space_group_IT_number"]
    print(PhaseName)
    PG = symmetry.get_point_group(int(numSG), True).name
    print(PG)
    array = nScoresOri[-1, :, :, :] # de base, les quaternions sont sur l'axe 1, on les veut sur l'axe 3

    height = len(array[0])
    width = len(array[0][0])

    listIPF = []

    # Sorti des cartes

    IPFim_X, xmap = IPF_Z(nScoresOri, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.xvector())
    IPFim_Y, xmap = IPF_Z(nScoresOri, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.yvector())
    IPFim_Z, xmap = IPF_Z(nScoresOri, PhaseName, PG, phases, nScores, Ipf_dir = Vector3d.zvector())

    # subplots

    f, axarr = plt.subplots(1,3) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    arr1 = axarr[0].imshow(IPFim_X, interpolation='none')
    arr2 = axarr[1].imshow(IPFim_Y, interpolation='none')
    arr3 = axarr[2].imshow(IPFim_Z, interpolation='none')

    arr1.axes.get_xaxis().set_visible(False)
    arr1.axes.get_yaxis().set_visible(False)

    arr2.axes.get_xaxis().set_visible(False)
    arr2.axes.get_yaxis().set_visible(False)

    arr3.axes.get_xaxis().set_visible(False)
    arr3.axes.get_yaxis().set_visible(False)

    axarr[0].set_title('IPF-X',fontsize = 50)
    axarr[1].set_title('IPF-Y',fontsize = 50)
    axarr[2].set_title('IPF-Z',fontsize = 50)

    plt.show()