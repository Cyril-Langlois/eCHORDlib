#! J:\Program Files\Python310\envs\Image\Scripts\python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:53:26 2023
@author: clanglois1
"""
import os

import eCHORDlib.Xallo as xa
from eCHORDlib import Symetry as sy
from eCHORDlib import Fct_profil_modification as fct
from eCHORDlib import general_functions as gf

from numba import jit
from pyquaternion import Quaternion
import os.path
import cupy as cp
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog, Tk
from configparser import ConfigParser
import tifffile as tf
import time

import scipy.io.wavfile
import scipy.signal
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt

def DBopen(DB):
    '''    
    Parameters
    ----------
    DB : TYPE format .h5
        DESCRIPTION. database CHORDv3 non modifiée

    Returns
    -------
    listLabelNames : TYPE Python list
        DESCRIPTION. Contient les clés d'accès aux datasets contenant les quaternions dans le fichier h5'
    listChunksNames : TYPE Python list
        DESCRIPTION. Contient les clés d'accès aux datasets contenant les profils dans le fichier h
    listChunkArrays : TYPE Python list
        DESCRIPTION. chaque objet de la liste est un des datasets récupérés dans le fichier h5 qui contient les profils
    listLabelArray : TYPE Python list
        DESCRIPTION.
    rawProfileLength : TYPE
        DESCRIPTION. chaque objet de la liste est un des datasets récupérés dans le fichier h5 qui contient les quaternions

    '''
    
    # lecture du fichier de profils theoriques test
    f = h5py.File(DB, 'r')
    
    listKeys = gf.get_dataset_keys(f)
    listGroups = gf.get_group_keys(f)
    
    # création des listes contenant les paths des profils théoriques et orientation dans la base    

    # check de la longueur des profils dans la base
    rawProfileLength = 0
    
    for i in listGroups:
        if "Sampling" in f[i].attrs.keys():
            rawProfileLength = int(f[i].attrs["Sampling"])
            
    if rawProfileLength == 0:
        rawProfileLength = 360 # valeur par défaut si Sampling non trouvé dans la base
    
    listLabelNames = []
    listChunksNames = []
    
    listChunkArrays = []
    listLabelArray = []
    
    for i in listKeys:
        if "LabelChunk" in i:
            listLabelNames.append(i)
        elif "DataChunk" in i:
            listChunksNames.append(i)

    for i in range(len(listChunksNames)):
        listChunkArrays.append(np.asarray(f[listChunksNames[i]]))
        listLabelArray.append(np.asarray(f[listLabelNames[i]]))

        # listChunkArray est une liste dont les éléments sont des tableaux Numpy
        # de 250_000 * rawProfileLength, le tout rangé en ligne, donc de dim 1
    
    return listLabelNames, listChunksNames, listChunkArrays, listLabelArray, rawProfileLength     

#%% Début ajout GDLH double indexation 

def Filtre_from_parameter(mapping,Seuil):
    Seuil = Seuil
    
    Filtre_mapping = np.zeros((len(mapping),len(mapping[0])))

    for i in range(0,len(mapping)):
        for j in range(0,len(mapping[0])):
            if mapping[i,j] >= Seuil:
                Filtre_mapping[i,j] = 0
            else:
                Filtre_mapping[i,j] = 1
                
    return Filtre_mapping,Seuil

def Grain_bdr_distance_filter(Distance_JDG,Seuil_KAM_distance):
    
    Seuil_KAM_distance = Seuil_KAM_distance
    Filtre_JDG = np.zeros((len(Distance_JDG),len(Distance_JDG[0])))

    for i in range(0,len(Distance_JDG)):
        for j in range(0,len(Distance_JDG[0])):
            if Distance_JDG[i,j] >= Seuil_KAM_distance:
                Filtre_JDG[i,j] = 1
            else:
                Filtre_JDG[i,j] = 0
                
    return Filtre_JDG,Seuil_KAM_distance

def KAM_DisOrientation(nScoresOri, SymQ):
    Best_quat = nScoresOri[-1,:,:,:] # Meilleurs Quaternion à analyser

    Ori1 = np.swapaxes(Best_quat, 0, 2)
    Map_filt_var = np.zeros((len(Ori1[0]),len(Ori1)))
    Map_disorientation = np.zeros((len(Ori1),len(Ori1[0])))

    for i in range(1,len(Ori1)-1):
        for j in range(1,len(Ori1[0])-1):
            ref_quat = Quaternion(Ori1[i,j,:]).inverse
            ref_quat1 = Quaternion(Ori1[i-1,j-1,:]).inverse
            ref_quat2 = Quaternion(Ori1[i-1,j,:]).inverse
            ref_quat3 = Quaternion(Ori1[i-1,j+1,:]).inverse
            ref_quat4 = Quaternion(Ori1[i,j-1,:]).inverse                 
            ref_quat5 = Quaternion(Ori1[i,j+1,:]).inverse
            ref_quat6 = Quaternion(Ori1[i+1,j-1,:]).inverse
            ref_quat7 = Quaternion(Ori1[i+1,j+1,:]).inverse
            ref_quat8 = Quaternion(Ori1[i+1,j,:]).inverse                   
                     
            Pseudo_Kernel_diso= (xa.disOfromQuatSymNoMat(ref_quat, ref_quat1, SymQ)[1] + xa.disOfromQuatSymNoMat(ref_quat, ref_quat2, SymQ)[1] + xa.disOfromQuatSymNoMat(ref_quat, ref_quat3, SymQ)[1]
                                  + xa.disOfromQuatSymNoMat(ref_quat, ref_quat4, SymQ)[1]+ xa.disOfromQuatSymNoMat(ref_quat, ref_quat4, SymQ)[1]+ xa.disOfromQuatSymNoMat(ref_quat, ref_quat6, SymQ)[1]
                                  + xa.disOfromQuatSymNoMat(ref_quat, ref_quat7, SymQ)[1]+ xa.disOfromQuatSymNoMat(ref_quat, ref_quat8, SymQ)[1])/8
            
            Map_disorientation[i,j]=Pseudo_Kernel_diso

    Map_disorientation_corr = np.rot90(Map_disorientation)
    Map_disorientation_corr = np.flipud(Map_disorientation_corr)

    del(ref_quat,ref_quat1,ref_quat2,ref_quat3,ref_quat4,ref_quat5,ref_quat6,ref_quat7,ref_quat8,Pseudo_Kernel_diso)

    return Map_filt_var,Map_disorientation_corr

def Masque_JDG_CHORD(Grain_bdr):
    Filtre_Gbdr = Grain_bdr == 0

    Grains = np.zeros((len(Filtre_Gbdr),len(Filtre_Gbdr[0])))

    for i in range(0,len(Grains)):
        for j in range(0,len(Grains[0])):
            if Filtre_Gbdr[i,j] == True:
                Grains[i,j] = 1
            else : 
                Grains[i,j] = 0 
    
    return Grains

def Masque_JDG_profil(rawImage_1):  
    Pr_length = len(rawImage_1)
    
    rawImage_roll1 = np.roll(rawImage_1,1, axis = 1)
    rawImage_roll2 = np.roll(rawImage_1,1, axis = 2)
    rawImage_roll3 = np.roll(rawImage_1,-1, axis = 1)
    rawImage_roll4 = np.roll(rawImage_1,-1, axis = 2)
    rawImage_roll5 = np.roll(rawImage_1,(1,1), axis = (1,2))
    rawImage_roll6 = np.roll(rawImage_1,(1,-1), axis = (1,2))
    rawImage_roll7 = np.roll(rawImage_1,(-1,1), axis = (1,2))
    rawImage_roll8 = np.roll(rawImage_1,(-1,-1), axis = (1,2))
    
    # Reshape des profils
    
    rawImage_reshape = np.reshape(np.swapaxes(rawImage_1,0,2),(-1,Pr_length))
    rawImage_roll1_reshape = np.reshape(np.swapaxes(rawImage_roll1,0,2),(-1,Pr_length)).T
    rawImage_roll2_reshape = np.reshape(np.swapaxes(rawImage_roll2,0,2),(-1,Pr_length)).T
    rawImage_roll3_reshape = np.reshape(np.swapaxes(rawImage_roll3,0,2),(-1,Pr_length)).T
    rawImage_roll4_reshape = np.reshape(np.swapaxes(rawImage_roll4,0,2),(-1,Pr_length)).T
    rawImage_roll5_reshape = np.reshape(np.swapaxes(rawImage_roll5,0,2),(-1,Pr_length)).T
    rawImage_roll6_reshape = np.reshape(np.swapaxes(rawImage_roll6,0,2),(-1,Pr_length)).T
    rawImage_roll7_reshape = np.reshape(np.swapaxes(rawImage_roll7,0,2),(-1,Pr_length)).T
    rawImage_roll8_reshape = np.reshape(np.swapaxes(rawImage_roll8,0,2),(-1,Pr_length)).T
    
    # Calcul de la distance entre la carte de rÃ©fÃ©rence et chaques cartes "rolls"
    
    Matmul1 = []
    Matmul2 = []
    Matmul3 = []
    Matmul4 = []
    Matmul5 = []
    Matmul6 = []
    Matmul7 = []
    Matmul8 = []
    
    for i in range(0,len(rawImage_reshape)):
        Var_1 = np.matmul(rawImage_reshape[i,:],rawImage_roll1_reshape[:,i])
        Var_2 = np.matmul(rawImage_reshape[i,:],rawImage_roll2_reshape[:,i])
        Var_3 = np.matmul(rawImage_reshape[i,:],rawImage_roll3_reshape[:,i])
        Var_4 = np.matmul(rawImage_reshape[i,:],rawImage_roll4_reshape[:,i])
        Var_5 = np.matmul(rawImage_reshape[i,:],rawImage_roll5_reshape[:,i])
        Var_6 = np.matmul(rawImage_reshape[i,:],rawImage_roll6_reshape[:,i])
        Var_7 = np.matmul(rawImage_reshape[i,:],rawImage_roll7_reshape[:,i])
        Var_8 = np.matmul(rawImage_reshape[i,:],rawImage_roll8_reshape[:,i])
        
        Matmul1.append(Var_1)
        Matmul2.append(Var_2)
        Matmul3.append(Var_3)
        Matmul4.append(Var_4)
        Matmul5.append(Var_5)
        Matmul6.append(Var_6)
        Matmul7.append(Var_7)
        Matmul8.append(Var_8)
    
    Matmul1 = np.vstack(Matmul1)
    Matmul2 = np.vstack(Matmul2)
    Matmul3 = np.vstack(Matmul3)
    Matmul4 = np.vstack(Matmul4)
    Matmul5 = np.vstack(Matmul5)
    Matmul6 = np.vstack(Matmul6)
    Matmul7 = np.vstack(Matmul7)
    Matmul8 = np.vstack(Matmul8)
    
    Matmul1 = np.reshape(Matmul1,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul2 = np.reshape(Matmul2,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul3 = np.reshape(Matmul3,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul4 = np.reshape(Matmul4,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul5 = np.reshape(Matmul5,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul6 = np.reshape(Matmul6,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul7 = np.reshape(Matmul7,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    Matmul8 = np.reshape(Matmul8,(len(rawImage_1[0][0]),len(rawImage_1[0])))
    
    Distance_JDG = (Matmul1+Matmul2+Matmul3+Matmul4+Matmul5+Matmul6+Matmul7+Matmul8)/8
    
    Distance_JDG = np.rot90(Distance_JDG, k=1, axes=(0, 1))
    Distance_JDG = np.flip(Distance_JDG)
    Distance_JDG = np.fliplr(Distance_JDG)
    Distance_JDG = Distance_JDG * 100
    
    Distance_hist_JDG = np.reshape(Distance_JDG,(-1,1))
    
    del(Var_1,Var_2,Var_3,Var_4,Var_5,Var_6,Var_7,Var_8)
    del(rawImage_roll1,rawImage_roll2,rawImage_roll3,rawImage_roll4,rawImage_roll5,rawImage_roll6,rawImage_roll7,rawImage_roll8)
    del(rawImage_roll1_reshape,rawImage_roll2_reshape,rawImage_roll3_reshape,rawImage_roll4_reshape,rawImage_roll5_reshape,rawImage_roll6_reshape,rawImage_roll7_reshape,rawImage_roll8_reshape)
    del(Matmul1,Matmul2,Matmul3,Matmul4,Matmul5,Matmul6,Matmul7,Matmul8)

    return Distance_JDG,Distance_hist_JDG

def reduction_reindexation(Map_filt,Grains):
# Filtres possibles : Map_filt(KAM-DisO d'orientation 1) / Filtre_mean_DisO_1(mean_DisO d'orientation 1)
# Masques possibles : Grains(grains issus de CHORD) / Filtre_JDG(JDG issus de la distance des profils avec leurs voisins)

    Filtre_X = Map_filt #Filtre des pixels à reindexer
    JDG_Y = Grains #Masque de réduction des données à réindexer
    
    Index_filtre = np.zeros((len(Filtre_X),len(Filtre_X[0])))
    
    for i in range(0,len(Index_filtre)):
        for j in range(0,len(Index_filtre[0])):
            if JDG_Y[i,j] == 0:
                Index_filtre[i,j] = 1
            else:
                Index_filtre[i,j] = Filtre_X[i,j]
    
    return Index_filtre

def extraction_reindexation(Index_filtre,rawImage_8bits):
# Les choix pour ré-indexer : Map_filt(KAM-DisOrientation orientaiton 1) / Filtre_mean_DisO_1(Mean_Orientation orientation 1)
# Grains (Masque grains issus de CHORD)/ Filtre_JDG(Masque JDG issu du calcul des distances des profils)
# Index_filtre (assemblage filtre + masque précédemment réalisé)

    Selected_filter = Index_filtre # Filtre sur la carte hybride Filtre + JDG

    # Creation de la stack ré-indexation
     
    List_scd_ind = []
    Recap_pos = []
    
    for i in range (0,len(Selected_filter)):
        for j in range (0,len(Selected_filter[0])):
            if Selected_filter[i,j] == 0:
                Var_pos=i,j
                Recap_pos.append(Var_pos)
                Var = rawImage_8bits[:,i,j]
                List_scd_ind.append(Var)
            
    del(Var)
            
    List_scd_ind = np.dstack(List_scd_ind)
    List_scd_ind = np.swapaxes(List_scd_ind, 0, 1)

    return List_scd_ind,Recap_pos

def Carte_final_reduced(nScoresOri_final,Map_filt_final):
# Les choix :  Map_filt_final(KAM-DisOrientation orientation final) / Filtre_mean_DisO_final(Mean_Orientation orientation final)
# Grains (Masque grains issus de CHORD)/ Filtre_JDG(Masque JDG issu du calcul des distances des profils)
# Index_final_filtre (assemblage filtre + masque précédemment réalisé)

    X = Map_filt_final
    
    nScoresOri_final_filt = np.zeros((len(nScoresOri_final),4,len(Map_filt_final),len(Map_filt_final[0])))
    
    for i in range(0,len(nScoresOri_final_filt[0][0])):
        for j in range(0,len(nScoresOri_final_filt[0][0][0])):
            if X[i,j] == 0 :
                nScoresOri_final_filt[:,:,i,j] = 0
            else :
                nScoresOri_final_filt[:,:,i,j] = nScoresOri_final[:,:,i,j]
                
    return nScoresOri_final_filt

def Carte_final_reduced2(nScoresOri_final, Map_filt_final, Grains):
# Les choix :  Map_filt_final(KAM-DisOrientation orientation final) / Filtre_mean_DisO_final(Mean_Orientation orientation final)
# Grains (Masque grains issus de CHORD)/ Filtre_JDG(Masque JDG issu du calcul des distances des profils)

    Filtre_final_X = Map_filt_final #Filtre 
    JDG_final_Y = Grains # Masque

    Index_final_filtre = np.zeros((len(Filtre_final_X),len(Filtre_final_X[0])))

    for i in range(0,len(Index_final_filtre)):
        for j in range(0,len(Index_final_filtre[0])):
            if JDG_final_Y[i,j] == 0:
                Index_final_filtre[i,j] = 0
            else:
                Index_final_filtre[i,j] = Filtre_final_X[i,j]
    
    return Index_final_filtre

def SaveFigure_Solo(Map_disorientation_corr, SaveFile, my_name = 'Nom.tiff',legend = 'legende',Titre = 'titre'):

    fig, ax = plt.subplots()
    legend = legend
    Titre = Titre
    SaveFile = SaveFile

    shw = ax.imshow(Map_disorientation_corr,cmap = 'plasma')
    ax.set_title(Titre)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
       
    Bar = plt.colorbar(shw, cax=cax)
    Bar.set_label(legend)
    ax.set_axis_off()

    my_name = my_name
    plt.savefig(os.path.join(SaveFile, my_name),dpi = 1000)

def SaveFigure_Duo(Map_disorientation_corr,Map_filt, SaveFile, my_name = 'Nom.tiff',legend1 = 'legende',Titre1 = 'titre',legend2 = 'legende',Titre2 = 'titre'):

    fig, (ax1,ax2) = plt.subplots(1,2)
    SaveFile = SaveFile
    legend1 = legend1
    Titre1 = Titre1
    my_name = my_name
    
    legend2 = legend2
    Titre2 = Titre2

    shw_diso = ax1.imshow(Map_disorientation_corr,cmap = 'plasma')
    ax1.set_title(Titre1)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    Bar1 = plt.colorbar(shw_diso, cax=cax)
    Bar1.set_label(legend1)
    ax1.set_axis_off()

    shw_map_filt = ax2.imshow(Map_filt,cmap = 'plasma')
    ax2.set_title(Titre2)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
   
    Bar2 = plt.colorbar(shw_map_filt, cax=cax)
    Bar2.set_label(legend2)
    ax2.set_axis_off()

    plt.savefig(os.path.join(SaveFile, my_name),dpi = 1000)  

# Fin ajout GDLH double indexation        

#%% classes et fonctions liées à l'indexation

class IndexationGPUderiv:
    '''
    Le programme prend en entrée :

    le tableau Numpy rawImage est constitué de :
        # en colonne (selon axe 0) les profils.
        # en axe 1, la dimension de l'axe 1 est le nombre de lignes de l'image (sa hauteur)
        # en axe 2, la dimension de l'axe 2 (en profondeur) correspond à la largeur de l'image
    
    savePath est le nom du répertoire où stocker le fichier de résultat (sans slash à la fin)
    
    Database et CIFfile sont des chemins vers les fichiers correspondants,
    à rentrer à l'instanciation de l'objet de la classe Indexation.
    nbSTACK est le nombre de profils de la stack que la carte est capable de prendre
    nbDB est le nb de profils de la base que la carte graphique peut prendre, 
    en conjonction avec nbSTACK
    
    dimPROF est la dimension d'un profil de la stack'
    
    
    Le but de la fonction est d'indexer sur GPU les profils de la stack qu'on lui donne
    
    à l'instanciation, l'indexation se met en route (grâce à __init__). La 
    fonction sauve le résultat dans un fichier h5 mais peut aussi retourner une stack
    de même taille avec les quaternions au lieu des profils (éventuellement avec les différents scores)
    '''    

    def __init__(self, Image, savePath, database, CIFfile, Workflow=[['Diff',0]], nScore = 5, normType = "centered euclidian", nbSTACK=20_000, nbDB = 20_000, dimPROF = 180):
    # def __init__(self, Image, savePath, database, CIFfile, nScore = 5, normType = "centered euclidian", nbDeriv = 0, filt="HLPF", pass_filter = "highpass", N_order = 1, Wn_order=0.1, nbSTACK=20_000, nbDB = 20_000, dimPROF = 180):

        print(database)        

        # instruction pour afficher les tableaux de façon lisible
        np.set_printoptions(suppress=True, precision=5)
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()
        
        with cp.cuda.Device(0):
            # mempool.set_limit(size=10*1024**3)
            self.mempool.set_limit(fraction=1.0)
            print(self.mempool.get_limit())      
        
        self.savePath = savePath   
       
        self.rawImage = Image    
        self.DB = database
        self.CIF = CIFfile

        # self.nbDeriv = nbDeriv        
        # self.filt = filt
        # self.N_order = N_order
        # self.Wn_order = Wn_order
        # self.pass_filter = pass_filter
        
        self.nbSTACK = nbSTACK
        self.nbDB = nbDB
        self.dimPROF = dimPROF
        self.nScore = nScore

        self.Workflow = Workflow
        
        
        # if normType == "euclidian" and nbDeriv > 0:
        #     print("Euclidian norm not compatible with profile differentiation")
        #     print("nbDeriv set to 0")
        #     nbDeriv = 0

        self.normType = normType
        
        self.dimExpProfiles = self.rawImage.shape[0]
        self.height = self.rawImage.shape[1]
        
        if len(self.rawImage.shape) < 3:
            self.width = 1
        else:
            self.width = self.rawImage.shape[2]
        
        self.listChunksNames = []
        self.listLabelNames = []
        
        self.listChunkArrays = []
        self.listLabelArray = []
        self.testArrayList = []

        self.maxDistList = []
        self.rawIndicesList = []
        self.miniChunk_finalList = []
        self.whichDataChunkList = []

        
        self.SymM = sy.get_proper_matrices_from_CIF(CIFfile)
        self.SymQ = sy.get_proper_quaternions_from_CIF(CIFfile)
        
        # ajuste les performances qui ont été calculées pour des profils de 180
        # si les profils exp font 360, alors la matrice GPU aura deux fois moins de lignes

        self.actualProfLength = 0
        self.expChunkNB = 0
    
    def runIndexation(self):

        self.dataPrepDiff()     
        self.nbSTACK = int(self.nbSTACK * self.dimPROF / self.actualProfLength)              
        self.expPrepDiff()        

        self.initIndexation()

        # if self.indexDiff:
            # self.indexationDiffspeed()
        # else:
        #     self.indexation()
        self.indexationDiffspeed()
        
        self.postIndexation()
        
        self.disOfinal()
        
        self.savingRes()
        
        self.savingMTEX()
        
    def runReIndexation(self,var):

        self.dataPrepDiff_ind2(var) 

        self.dataPrepDiff()  
        self.nbSTACK = int(self.nbSTACK * self.dimPROF / self.actualProfLength)              
        self.expPrepDiff()
        
        self.initIndexation()

        self.indexationDiffspeed()
        
        self.postIndexation()
        
        self.disOfinal()
        
        self.savingRes()
        
        self.savingMTEX()
            
    def dataPrepDiff(self):
        
        self.listLabelNames, self.listChunksNames, self.listChunkArrays, self.listLabelArray, self.rawProfileLength = DBopen(self.DB)
        
        # détermination du facteur de réduction
        actualRatio = self.rawProfileLength / self.dimExpProfiles
        if actualRatio < 1.0:
            print("increase theoretical profile length")
            quit()
        elif actualRatio == 1.0: # les profils theo et exp ont la même longueur
            reductionFactor = 1
        else:
            if (self.rawProfileLength % self.dimExpProfiles) == 0:
                reductionFactor = np.floor_divide(self.rawProfileLength, self.dimExpProfiles)
            else:
                "dim exp profiles must divide dim theoretical profiles"
                quit()
                
        self.actualProfLength = int(self.rawProfileLength / reductionFactor)
        
        self.listChunkArraysDiff = []
        
        for i in range(len(self.listChunkArrays)):
            self.a = fct.downSampleProfiles(self.listChunkArrays[i], reductionFactor) # réduction de 360 à 180 mais profils toujours en ligne (1D array)
            self.listChunkArrays[i] = self.a
            self.b = fct.reshapeProfilesInLine(self.a, self.actualProfLength)
            # c = Profile_modifier(b, axProf = 1, filt = self.filt, N_order = self.N_order, Wn_order = self.Wn_order, nDeriv = self.nbDeriv, pass_filter = self.pass_filter)
            self.c = fct.Profile_modifier(self.b, self.Workflow, axProf = 1)
            self.listChunkArraysDiff.append(self.c)
            
    def dataPrepDiff_ind2(self,var):
        
        self.listLabelNames = var.listLabelNames
        self.listChunksNames = var.listChunksNames
        self.listChunkArrays = var.listChunkArrays
        self.listLabelArray = var.listLabelArray
        self.rawProfileLength = var.rawProfileLength
        
        # détermination du facteur de réduction
        actualRatio = self.rawProfileLength / self.dimExpProfiles
        if actualRatio < 1.0:
            print("increase theoretical profile length")
            quit()
        elif actualRatio == 1.0: # les profils theo et exp ont la même longueur
            reductionFactor = 1
        else:
            if (self.rawProfileLength % self.dimExpProfiles) == 0:
                reductionFactor = np.floor_divide(self.rawProfileLength, self.dimExpProfiles)
            else:
                "dim exp profiles must divide dim theoretical profiles"
                quit()
                
        self.actualProfLength = int(self.rawProfileLength / reductionFactor)
        
        self.listChunkArraysDiff = []
        
        self.a = var.a
        self.b = var.b
        self.c = var.c

    def expPrepDiff(self):
        rawImage2D = self.rawImage.reshape(self.dimExpProfiles, self.height * self.width)
        
        # diffImage2D = Profile_modifier(rawImage2D, axProf = 0, filt = self.filt, N_order = self.N_order, Wn_order = self.Wn_order, nDeriv = self.nbDeriv, pass_filter = self.pass_filter)
        diffImage2D = fct.Profile_modifier(rawImage2D, self.Workflow, axProf = 0)
        self.expChunkNB = np.floor_divide(self.height * self.width, self.nbSTACK)
        self.remainSTACK = (self.height * self.width) % self.nbSTACK
        
        
        # pour les profils test, il y aura un nombre expChunkNB de chunks contenant nbSTACK*factorGPU pixels
        # il restera à passer une matrice de remainSTACK pixels (ceux de la fin de l'axe 1)
        
        i = 0
        for i in range(self.expChunkNB):
            self.testArrayList.append(diffImage2D[:,i*self.nbSTACK:(i+1)*self.nbSTACK])
        
        self.remain = 0
        if self.remainSTACK != 0:
            self.remain = 1
            self.testArrayList.append(diffImage2D[:, -self.remainSTACK:])
        print("fin expPrepDiff")
    
    def initIndexation(self):
        

        self.dbChunks = self.nbDB # 1/8 des profils d'un DataChunk. 62_500 sur UltraCalculus, 25_000 sur ordi perso
        

        self.loopDB = int(np.floor_divide(len(self.listChunkArrays[0]) / self.actualProfLength, self.dbChunks))
        print(f"loopDB = {self.loopDB}")
        
        if self.nScore > len(self.listChunksNames):
            print("nScores doit être inférieur au nb de Datachunks")
            self.nScore = len(self.listChunksNames)
            print(f"nouveau nScores : {self.nScore}")
        print("fin initIndexation")

    def indexationDiffspeed(self):
        print("enter indexationDiff")
        # initialisation des listes résultats
        
        maxDistList = []
        rawIndicesList = []
        miniChunk_finalList = []
        whichDataChunkList = []

        for y in range(self.expChunkNB + 1 * self.remain): # +1 pour prendre en compte le testChunk avec le reste
        # for y in range(2):
            normedGPUtest = cp.array(self.testArrayList[y])
                
            
            print(f"\nchunk test : #{y} / {self.expChunkNB}")
            print("________________________________________")
        
            nbSTACKcurr =  len(self.testArrayList[y][0])  
        
            distDataChunk = np.zeros((len(self.listChunksNames), nbSTACKcurr))
            indDataChunk = np.zeros((len(self.listChunksNames), nbSTACKcurr))
            miniChunkInd = np.zeros((len(self.listChunksNames), nbSTACKcurr))
            
            
            for j in range(len(self.listChunksNames)):

                
                # on charge un chunk entier  en ligne que l'on remet en matrice

                listDist = np.zeros((self.loopDB, nbSTACKcurr))
                listInd = np.zeros((self.loopDB, nbSTACKcurr))
                print(f"chunk DB : #{j} out of {len(self.listChunksNames)-1}")
                
                for k in range(self.loopDB): # on traite le DataChunk courant par petits bouts
                # for k in range(2): # on traite les deux premiers minichunk du DataChunk j
                    print(f"     miniChunkDB : #{k} out of {self.loopDB-1}")
                    # on charge dans la GPU un nouveau chunk de profils de la base            
                    normedGPU = cp.array(self.listChunkArraysDiff[j][k*self.dbChunks:(k+1)*self.dbChunks, :])

                    # calcul du tableau de distances
                    
                    distances = cp.matmul(normedGPU, normedGPUtest)
                    # dist = cp.asnumpy(distances)
                    
                    del normedGPU

                    self.mempool.free_all_blocks()
                    self.pinned_mempool.free_all_blocks()

                    listDist[k, :] = cp.asnumpy(cp.max(distances, axis=0))
                    listInd[k,:] = cp.asnumpy(np.argmax(distances, axis=0))
                    del distances            
                    # on libère la mémoire

                    # del dist
            
                    self.mempool.free_all_blocks()
                    self.pinned_mempool.free_all_blocks()
                    
                maxInd = np.argsort(listDist, axis=0)
                miniChunkInd[j, :] = maxInd[-1, :]
                distDataChunk[j, :] = np.take_along_axis(listDist, maxInd, axis=0)[-1, :]
                indDataChunk[j, :] = np.take_along_axis(listInd, maxInd, axis=0)[-1, :]
                
            maxInd_array = cp.argsort(distDataChunk, axis=0)
            maxInd_array = cp.asnumpy(maxInd_array)
            
            self.maxDistList.append(np.take_along_axis(distDataChunk, maxInd_array, axis=0)[-self.nScore:, :])
            self.rawIndicesList.append(np.take_along_axis(indDataChunk, maxInd_array, axis=0)[-self.nScore:, :])
            self.miniChunk_finalList.append(np.take_along_axis(miniChunkInd, maxInd_array, axis=0)[-self.nScore:, :])
            self.whichDataChunkList.append(maxInd_array[-self.nScore:, :])
        
        del normedGPUtest
        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()
        print(self.mempool.get_limit())

        print("exit indexationDiffspeed")

    def postIndexation(self):
        print("enter post-indexation")
        print("nscore = ", self.nScore)
        indexedStackList = []
        oriList = []
        Pr_list = []
        
        for y in range(self.expChunkNB + 1 * self.remain):
            indexedStackList.append(np.zeros((self.nScore, self.actualProfLength, len(self.testArrayList[y][0, :]) )))
            oriList.append(np.zeros((self.nScore, 4, len(self.testArrayList[y][0, :])))) # 4 because Quaternions 
        
        for n in range(self.expChunkNB + 1 * self.remain):
            
            for iTP in range(len(self.testArrayList[n][0, :])):
                
                for i in range(self.nScore):
                    DataChunk = fct.reshapeProfilesInLine(self.listChunkArrays[self.whichDataChunkList[n][i, iTP]], self.actualProfLength)
                    DataOri = fct.reshapeProfilesInLine(self.listLabelArray[self.whichDataChunkList[n][i, iTP]], 4)
                    
                    indInDataChunk = int(self.rawIndicesList[n][i, iTP] + self.miniChunk_finalList[n][i, iTP] * self.dbChunks)
                    
                    Pr_list.append(indInDataChunk)
                    
                    ori = DataOri[indInDataChunk, :]
        
                    profile = DataChunk[indInDataChunk, :]
                    
                    if self.normType == "euclidian":
                        profile = fct.normMatProfiles(profile, 0)
                    elif self.normType == "centered euclidian":
                        profile = fct.centeredEuclidianNorm(profile, 0)            
                    
                    indexedStackList[n][i, :, iTP] = profile
                    oriList[n][i, :, iTP] = ori
        
                    
                    del profile  
                self.mempool.free_all_blocks()
                self.pinned_mempool.free_all_blocks()           
        
        # concaténation
        
        nScoresStack = np.concatenate(indexedStackList, axis = -1)
        #Ajout GLDH
        
        nScoresDist = np.concatenate(self.maxDistList, axis = -1)
        nScoresOri = np.concatenate(oriList, axis = -1)
        
        whichDataChunkList = np.concatenate(self.whichDataChunkList,axis=1)
        Ref_Pr_list = np.reshape(np.vstack(Pr_list),(-1,self.nScore)).T
        # N° du profil théorique pour chaque pixel de l'image et pour chaque score
        Ref_Pr_list = Ref_Pr_list+(whichDataChunkList*self.loopDB*self.dbChunks)
        # N° du profil théorique pour chaque score (suivant X et Y)
        self.Ref_Pr_list2 = np.reshape(Ref_Pr_list,(self.nScore,self.height, self.width))
        
        self.nScoresStack = nScoresStack.reshape((self.nScore, self.actualProfLength, self.height, self.width))
        print("Shape nScoresStack")
        print(self.nScoresStack.shape)
                        
        self.nScoresDist = nScoresDist.reshape((self.nScore, self.height, self.width))
        self.nScoresOri = nScoresOri.reshape((self.nScore, 4, self.height, self.width))
        
        # Ajout GDLH pour récupération des profils avec traitement ==> A améliorer 
        
        self.Treatment_theo_prof = self.nScoresStack
        self.Treatment_theo_prof = self.Treatment_theo_prof.reshape((self.actualProfLength, self.height * self.width * self.nScore))
        self.Treatment_theo_prof = fct.Profile_modifier(self.Treatment_theo_prof, self.Workflow, axProf = 0)
        self.Treatment_theo_prof = self.Treatment_theo_prof.reshape((self.nScore, self.actualProfLength, self.height , self.width))
        
        # if self.Workflow[0][1] == 1:
        #     self.Treatment_theo_prof = self.Treatment_theo_prof.reshape((self.nScore, self.actualProfLength, self.height , self.width))
        # else : 
        #     self.Treatment_theo_prof = self.Treatment_theo_prof.reshape((self.nScore, self.actualProfLength + 1, self.height , self.width))

        # Ajout GDLH pour récupération des profils expérimentaux avec traitement ==> A améliorer 
        self.Exp_prof_treatment = np.hstack(self.testArrayList)
        self.Exp_prof_treatment = self.Exp_prof_treatment.reshape((self.actualProfLength , self.height , self.width))
        
        # if self.Workflow[0][1] == 1:    
        #     self.Exp_prof_treatment = self.Exp_prof_treatment.reshape((self.actualProfLength , self.height , self.width))
        # else :
        #     self.Exp_prof_treatment = self.Exp_prof_treatment.reshape((self.actualProfLength + 1, self.height , self.width))
        
        # self.Exp_prof_treatment = fct.Profile_modifier(self.Exp_prof_treatment, self.Workflow, axProf =0)
        
        # if self.Workflow[0][1] == 1:    
        #     self.Exp_prof_treatment = self.Exp_prof_treatment.reshape((self.actualProfLength , self.height , self.width))
        # else :
        #     self.Exp_prof_treatment = self.Exp_prof_treatment.reshape((self.actualProfLength + 2, self.height , self.width))
        # Ajout GLDH
       
        if self.normType == "euclidian":
            rawImage = fct.normMatProfiles(self.rawImage, 0)
        elif self.normType == "centered euclidian":
            rawImage = fct.centeredEuclidianNorm(self.rawImage, 0)
        
        self.rawImage = rawImage
        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()
        print("end post-indexation")


    def disOfinal(self):

        print("start disO")      
        self.nScoresDisO = np.zeros((self.nScore, self.height, self.width))
        
        for i in range(self.height):
            for j in range(self.width):
                
                refO = Quaternion(self.nScoresOri[-1, :, i, j]).inverse
                
                for k in range(self.nScore):
                    ori = Quaternion(self.nScoresOri[k, :, i, j]).inverse
                    
                    self.nScoresDisO[k, i, j] = xa.disOfromQuatSymNoMat(refO, ori, self.SymQ)[1]
        
        self.meanDisO = np.mean(self.nScoresDisO, axis = 0)
        
        print(self.nScoresDisO.shape)
        print("end disO")

    def savingRes(self):
        
        print("start saving")

        ti = time.strftime("%Y-%m-%d__%Hh-%Mm")
        
        indexSTACK = h5py.File(self.savePath + '\indexScores_'+ ti + '.hdf5', 'a')
        
        group = indexSTACK.create_group('indexation')
        group.create_dataset(name='nScoresStack', data=self.nScoresStack)
        group.create_dataset(name='Treatment_theo_prof', data=self.Treatment_theo_prof) #Profil théo modifiés
        group.create_dataset(name='rawImage', data=self.rawImage)
        group.create_dataset(name='nScoresDist', data=self.nScoresDist)
        group.create_dataset(name='nScoresDisO', data=self.nScoresDisO)
        group.create_dataset(name='nScoresOri', data=self.nScoresOri)
        group.create_dataset(name='meanDisO', data=self.meanDisO)
        group.create_dataset(name='Ref_Pr_list3', data=self.Ref_Pr_list2)
        group.create_dataset(name='testArrayList', data=self.Exp_prof_treatment) #Profil expé modifiés
                 
        group.attrs.create("profile length", self.actualProfLength)
        group.attrs.create("dbChunks", self.dbChunks)
        group.attrs.create("height", self.height)
        group.attrs.create("width", self.width)
        group.attrs.create("nScores", self.nScore)
        group.attrs.create("CIF path", self.CIF)
        group.attrs.create("stack path", self.savePath)
        group.attrs.create("database path", self.DB)
        group.attrs.create("normalization before indexation", self.normType)
        group.attrs.create("metric for Indexation", "cosine")
        group.attrs.create("nbSTACK", self.nbSTACK)
        group.attrs.create("nbDB", self.nbDB)
        
        indexSTACK.flush()
        indexSTACK.close()
        print("end saving")    

    def savingMTEX(self):
        
        Quat = self.nScoresOri[-1,:,:,:]
        x = len(Quat[0])
        y = len(Quat[0][0])

        ti = time.strftime("%Y-%m-%d__%Hh-%Mm")

        with open(self.savePath + '\indexGPU_'+ ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:

            for i in range(x):
                for j in range(y):

                    index = 1    
                    if Quat[0,i,j] ==0 :
                        index = 0
                    file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                    '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n')

#%% Définitions de la ré-indexation 
# Création des dossiers de rangement des filtres / masques / résultats tifffile
    def Reind_location(self):
        
        window = Tk()
        window.wm_attributes('-topmost', 1)
        window.withdraw()

        self.current_directory = filedialog.askdirectory() # Dossier ou vont être créer les nouveaux dossiers

        self.dirFilter = "Filtre"
        self.DirFilter = os.path.join(self.current_directory, self.dirFilter)
        os.mkdir(self.DirFilter) # Création du dossier "Indexation1"

        self.dirMask = "Masque"
        self.DirMask = os.path.join(self.current_directory, self.dirMask)
        os.mkdir(self.DirMask) # Création du dossier "Indexation2"
        
        self.dir_tiff = "Res_tiff"
        self.Dir_tiff = os.path.join(self.current_directory, self.dir_tiff)
        os.mkdir(self.Dir_tiff) # Création du dossier "Indexation2"

        del(self.dirFilter,self.dirMask, self.dir_tiff)
        
# Filtre sur meanDisO
    def View_meanDisO(self,meanDisO, my_name = 'meanDisO_Orientation1.tiff'):
        # Filtre orientation 1 : Création du filtre à partir de meanDisO
        SaveFigure_Solo(meanDisO, self.DirFilter, my_name, legend = 'Mean Misorientation (°)', Titre = 'Mean misorientation map')

    def Filter_meanDisO(self, meanDisO, Seuil_MeanDisO = 2, my_name ='Subplot_Mean_DisO_and_Filter.tiff'):
        # Calcul du filtre sur les mean disorientation
        [self.Filtre_mean_DisO_1,Seuil_mean_DisO] = Filtre_from_parameter(meanDisO,Seuil_MeanDisO) # Calcul du Mean_DisOrientation 1
        SaveFigure_Duo(meanDisO,self.Filtre_mean_DisO_1, self.DirFilter, my_name, legend1 = 'Mean Misorientation (°)',Titre1 = 'Mean Misorientation map',legend2 = 'filtre (0 = exclude ; 1=keep)',Titre2 = 'Seuil de sélection par Mean Miso(°) =%i' %Seuil_mean_DisO)

# Filtre sur KAM disorientation
    def View_DisOKAM(self, nScoresOri, my_name = 'KAM_Misorientation_map.tiff'):
        # Calcul du KAM sur les désorientations
        [Map_filt,Map_disorientation_corr] = KAM_DisOrientation(nScoresOri, self.SymQ) # Calcul du KAM d'orientation 1
        # Sauvegarde de la figure  
        SaveFigure_Solo(Map_disorientation_corr, self.DirFilter, my_name, legend = 'Misorientation (°)',Titre = 'KAM misorientation map')
        self.Map_disorientation_corr = Map_disorientation_corr
        self.Map_filt = Map_filt

    def Filter_DisOKAM(self, Map_disorientation_corr, Seuil = 5, my_name = 'KAM_Misorientation_Filter.tiff', my_name1 = 'Subplot_KAM_Misorientation_&_Filter.tiff'):
        # Création du filtre sur le KAM des désorientations
        [self.Map_filt,Seuil_KAM_ori1] = Filtre_from_parameter(Map_disorientation_corr, Seuil) # Calcul du KAM d'orientation final
        # Sauvegarde des figures : filtre KAM et subplot Misorientation map + filtre KAM
        SaveFigure_Solo(self.Map_filt, self.DirFilter, my_name, legend = 'filtre (0 = exclude ; 1=keep)',Titre = 'Seuil de sélection (°) =%i' %Seuil_KAM_ori1)
        SaveFigure_Duo(Map_disorientation_corr,self.Map_filt, self.DirFilter, my_name1, legend1 = 'Misorientation (°)',Titre1 = 'KAM misorientation map',legend2 = 'filtre (0 = exclude ; 1=keep)',Titre2 = 'Seuil de sélection (°) =%i' %Seuil_KAM_ori1)

# Masque sur JDG CHORD
    def import_CHORD_GrainBDR(self):
        
        f, _, _, listKeys = gf.openH5file()
        
        for i in listKeys:
            if "MapData" in i:
                self.Grain_bdr = np.asarray(f[i])
        
        self.Grain_bdr = np.rot90(self.Grain_bdr, k=1, axes=(0, 1))
        self.Grain_bdr = np.flip(self.Grain_bdr)
        self.Grain_bdr = np.fliplr(self.Grain_bdr)

    def Masque_CHORD(self):
        # Création du masque des JDG CHORD
        self.Grains = Masque_JDG_CHORD(self.Grain_bdr) 
        # Sauvegarde de la figure
        SaveFigure_Solo(self.Grains, self.DirMask, my_name = 'CHORD grain boundaries.tiff',legend = 'filtre (0 = Grain boudaries ; 1 = Grain',Titre = 'CHORD Grain boundaries') 

# Masque sur JDG distance des profils
    def View_Distance(self):
        # Visualisation du KAM distance
        self.rawImage_norm = fct.centeredEuclidianNorm(self.rawImage, 0)
        [self.Distance_JDG,Distance_hist_JDG] = Masque_JDG_profil(self.rawImage_norm)
        # Sauvegarde de la figure
        SaveFigure_Solo(self.Distance_JDG, self.DirMask, my_name = 'Distance_map_grain_boundaries.tiff',legend = 'Distance',Titre = 'distance map')

    def Filter_Distance(self, Seuil_KAM_distance = 0.93):
        [self.Filtre_JDG,Seuil_KAM_distance] = Grain_bdr_distance_filter(self.Distance_JDG,Seuil_KAM_distance)
        SaveFigure_Solo(self.Filtre_JDG, self.DirMask, my_name = 'Distance map filter.tiff', legend = 'filtre (0 = Grain boudaries ; 1 = Grain',Titre = 'Seuil de sélection par distance =%i' %Seuil_KAM_distance)
        SaveFigure_Duo(self.Distance_JDG,self.Filtre_JDG, self.DirMask, my_name = 'Subplot_Distance_map and filter.tiff',legend1 = 'Distance',Titre1 = 'distance map',legend2 = 'filtre (0 = Grain boudaries ; 1 = Grain',Titre2 = 'Seuil de sélection par distance =%i' %Seuil_KAM_distance)

# Ajout d'un masque sur la map filtre pour réduire le nombre de point à ré-indexer  
    def reduc_indexation2(self,Filtre_mean_DisO_1,Filtre_JDG):
        self.Index_filtre = reduction_reindexation(Filtre_mean_DisO_1,Filtre_JDG) #Filtre,Masque)
        # Filtres possibles : Map_filt(KAM-DisO d'orientation 1) / Filtre_mean_DisO_1(mean_DisO d'orientation 1)
        # Masques possibles : Grains(grains issus de CHORD) / Filtre_JDG(JDG issus de la distance des profils avec leurs voisins)
        SaveFigure_Solo(self.Index_filtre, self.DirFilter, my_name = 'Pixel à ré-indexer (moins JDG).tiff',legend = 'filtre (0 = exclude ; 1=keep)',Titre = 'Filtre + masque')

# Extraction des points à ré-indexer
    def import_rawImage_8bits(self):
        
        StackLoc, _ = gf.getFilePathDialog("série d'images à indexer' (*.tiff)")
        self.rawImage_8bits = tf.TiffFile(StackLoc).asarray()

    def Reind_extraction(self,Index_filtre):
        [self.List_scd_ind,self.Recap_pos] = extraction_reindexation(Index_filtre,self.rawImage_8bits)
        # Open a file dialog window to save the adjusted TIFF stack
        output_List_scd_ind = filedialog.asksaveasfilename(title="Save the second indexation TIFF stack", filetypes=[("TIFF Files", "*.tif")])
        # Save the adjusted stack as a new TIFF file
        tf.imwrite(output_List_scd_ind, self.List_scd_ind)        

    def Merge_data(self, nScoresOri_1, meanDisO_1, nScoresDisO_1, nScoresDist_1, nScoresStack_1, nScoresOri_2, meanDisO_2, nScoresDisO_2, nScoresDist_2, nScoresStack_2):
        # Création de la cartographie finale hybride, issue des deux indexations
        self.nScoresOri_1f = nScoresOri_1
        self.nScoresOri_2f = nScoresOri_2
        
        self.nScoresOri_final =self. nScoresOri_1f
        self.meanDisO_final = meanDisO_1
        self.nScoresDisO_final = nScoresDisO_1
        self.nScoresDist_final = nScoresDist_1
        self.nScoresStack_final = nScoresStack_1
        
        # Remplacement des quaternions par les nouveaux (indexation 2) pour les positions de pixel considÃ©rÃ©s
                   
        for i in range(0,len(self.Recap_pos)):
            Varx = self.Recap_pos[i][0]
            Vary = self.Recap_pos[i][1]
            self.nScoresOri_final[:,:,Varx,Vary]  =  self.nScoresOri_2f[:,:,0,i]
            self.meanDisO_final[Varx,Vary]  =  meanDisO_2[0,i]
            self.nScoresDisO_final[:,Varx,Vary]  =  nScoresDisO_2[:,0,i]
            self.nScoresDist_final[:,Varx,Vary]  =  nScoresDist_2[:,0,i]
            self.nScoresStack_final[:,:,Varx,Vary]  =  nScoresStack_2[:,:,0,i]
        
        del(Varx,Vary)

# Creation de la carte avec pixels filtrés (filtre ou masque)
    def Map_finale(self, nScoresOri_final, Map_filt_final):
        self.nScoresOri_final_filt = Carte_final_reduced(nScoresOri_final, Map_filt_final)

# Creation de la carte avec pixels filtrés (2 filtres ; 2 masques ; 1 filtre 1 masque)
    def Map_finale2(self, nScoresOri_final, Map_filt_final, Grains, my_name = 'Pixel exclus finaux.tiff'):
        self.Index_final_filtre = Carte_final_reduced2(nScoresOri_final, Map_filt_final, Grains)
        SaveFigure_Solo(self.Index_final_filtre, self.DirFilter, my_name,legend = 'filtre (0 = exclude ; 1=keep)',Titre = 'Filtre + masque final')

#  Sauvegardes finales
#  Sauvegarde h5 (sans filtre apposé)
    def savingRes_finale(self):
        
        print("start saving final orientation")

        ti = time.strftime("%Y-%m-%d__%Hh-%Mm")
        
        indexSTACK = h5py.File(self.savePath + '\indexScores_final'+ ti + '.hdf5', 'a')
        
        group = indexSTACK.create_group('indexation')
        group.create_dataset(name='nScoresStack', data=self.nScoresStack_final)
        group.create_dataset(name='Treatment_theo_prof', data=self.Treatment_theo_prof) #Profil théo modifiés
        group.create_dataset(name='rawImage', data=self.rawImage)
        group.create_dataset(name='nScoresDist', data=self.nScoresDist_final)
        group.create_dataset(name='nScoresDisO', data=self.nScoresDisO_final)
        group.create_dataset(name='nScoresOri', data=self.nScoresOri_final)
        group.create_dataset(name='meanDisO', data=self.meanDisO_final)
        group.create_dataset(name='Ref_Pr_list3', data=self.Ref_Pr_list2)
        group.create_dataset(name='testArrayList', data=self.Exp_prof_treatment) #Profil expé modifiés
                 
        group.attrs.create("profile length", self.actualProfLength)
        group.attrs.create("dbChunks", self.dbChunks)
        group.attrs.create("height", self.height)
        group.attrs.create("width", self.width)
        group.attrs.create("nScores", self.nScore)
        group.attrs.create("CIF path", self.CIF)
        group.attrs.create("stack path", self.savePath)
        group.attrs.create("database path", self.DB)
        group.attrs.create("normalization before indexation", self.normType)
        group.attrs.create("metric for Indexation", "cosine")
        group.attrs.create("nbSTACK", self.nbSTACK)
        group.attrs.create("nbDB", self.nbDB)
        
        indexSTACK.flush()
        indexSTACK.close()
        print("end saving") 

    def savingMTEX_finale(self):
        
        Quat = self.nScoresOri_final[-1,:,:,:]
        x = len(Quat[0])
        y = len(Quat[0][0])

        ti = time.strftime("%Y-%m-%d__%Hh-%Mm")

        with open(self.savePath + '\indexGPU_final_'+ ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:

            for i in range(x):
                for j in range(y):

                    index = 1    
                    if Quat[0,i,j] ==0 :
                        index = 0
                    file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                    '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n')

    def savingMTEX_finale_filt(self):
        
        Quat = self.nScoresOri_final_filt[-1,:,:,:]
        x = len(Quat[0])
        y = len(Quat[0][0])

        ti = time.strftime("%Y-%m-%d__%Hh-%Mm") 

        with open(self.savePath + '\indexGPU_final_filt_'+ ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:

            for i in range(x):
                for j in range(y):

                    index = 1    
                    if Quat[0,i,j] ==0 :
                        index = 0
                    file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                    '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n')


class preReIndexation:
    def __init__(self):
        
        self.StackLoc, self.StackDir = gf.getFilePathDialog("série d'images à indexer' (*.tiff)")
        self.Stack = tf.TiffFile(self.StackLoc[0]).asarray()
        
# Fin de la partie de ré-indexation 

#%% Pre-indexation

class preIndexation:
    def __init__(self):
       
        self.StackLoc, self.StackDir = gf.getFilePathDialog("série d'images à indexer (*.tiff)")
        self.Stack = tf.TiffFile(self.StackLoc[0]).asarray()

        filepath,  self.CifDir = gf.getFilePathDialog('CIF selection')
        # self.CifLoc, self.CifDir = gf.getFilePathDialog('CIF selection')
        self.CifLoc = filepath[0]
        
        filepath, self.DatabaseDir = gf.getFilePathDialog('theoretical test profiles (*.crddb)')
        self.DatabaseLoc = filepath[0]
        # self.DatabaseLoc, self.DatabaseDir = gf.getFilePathDialog('theoretical test profiles (*.crddb)') 
        
        
        # self.StackLoc, self.StackDir = gf.getFilePathDialog("série d'images à indexer' (*.tiff)")
        # self.Stack = tf.TiffFile(self.StackLoc).asarray()

        # self.CifLoc, self.CifDir = gf.getFilePathDialog('CIF selection')
        
        # self.DatabaseLoc, self.DatabaseDir = gf.getFilePathDialog('theoretical test profiles (*.crddb)')
 
#%%
class Old_indexation(IndexationGPUderiv):
    ''' Le programme vient récuperer un ancien fichier h5 pour créer un objet 
    indexation qui puisse être utilisé pour la partie ré-indexation  '''
    
    def __init__(self, preInd):

        f, _, _, listKeys = gf.openH5file()
        
        for i in listKeys:
            if "meanDisO" in i:
                self.meanDisO = np.asarray(f[i])
            elif "nScoresDisO" in i:
                self.nScoresDisO = np.asarray(f[i])
            elif "nScoresDist" in i:
                self.nScoresDist = np.asarray(f[i])
            elif "nScoresStack" in i:
                self.nScoresStack = np.asarray(f[i])
            elif "nScoresOri" in i:
                self.nScoresOri = np.asarray(f[i])
            elif "rawImage" in i:
                self.rawImage = np.asarray(f[i])   
            elif "Treatment_theo_prof" in i:
                self.Treatment_theo_prof = np.asarray(f[i])   
       
        nScore = len(self.nScoresOri)
        
        IndexationGPUderiv.__init__(self, preInd.Stack, preInd.StackDir, preInd.DatabaseLoc, preInd.CifLoc, Workflow=[['Diff',0]], nScore = nScore, normType = "centered euclidian", nbSTACK=10_000, nbDB = 10_000, dimPROF = 180)
        
        self.dataPrepDiff()