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

from pyquaternion import Quaternion

import cupy as cp
import h5py
import numpy as np

import tifffile as tf
import time




#%% classes et fonctions liées à l'indexation

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

        print(database)
        print("editable library")
        

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

        self.nbSTACK = nbSTACK
        self.nbDB = nbDB
        self.dimPROF = dimPROF
        self.nScore = nScore

        self.Workflow = Workflow
        
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
                    
                    distances = cp.matmul(normedGPU, normedGPUtest)

                    del normedGPU

                    self.mempool.free_all_blocks()
                    self.pinned_mempool.free_all_blocks()

                    listDist[k, :] = cp.asnumpy(cp.max(distances, axis=0))
                    listInd[k,:] = cp.asnumpy(np.argmax(distances, axis=0))
                    del distances
            
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
        

        # Ajout GDLH pour récupération des profils expérimentaux avec traitement ==> A améliorer 
        self.Exp_prof_treatment = np.hstack(self.testArrayList)
        self.Exp_prof_treatment = self.Exp_prof_treatment.reshape((self.actualProfLength , self.height , self.width))
        
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
        sep = os.sep
        
        # path = os.path.normpath(self.savePath + '\indexScores_'+ ti + '.hdf5')
        # indexSTACK = h5py.File(path, 'a')
        indexSTACK = h5py.File(self.savePath + sep + 'indexScores_'+ ti + '.hdf5', 'a')
        
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
        sep = os.sep

        with open(self.savePath + sep + 'indexGPU_'+ ti + '.quatCHORDv3-CTFxyConv.txt', 'w') as file:

            for i in range(x):
                for j in range(y):

                    index = 1    
                    if Quat[0,i,j] ==0 :
                        index = 0
                    file.write(str(index) + '\t' + str(j) + '\t' + str(i) + '\t' + str(Quat[0, i, j]) +
                    '\t' + str(Quat[1, i, j])  + '\t' + str(Quat[2, i, j])  + '\t' + str(Quat[3, i, j]) + '\n')

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
