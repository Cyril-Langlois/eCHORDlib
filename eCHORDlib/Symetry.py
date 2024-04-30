# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:09:52 2020

@author: Cyril Langlois
"""


import os

sep = os.sep

from LibrairiesCyril import Xallo as xa
import numpy as np
import Dans_Diffraction as dif


# instruction pour afficher les atbleaux de façon lisible
np.set_printoptions(suppress=True, precision=2)

def get_proper_matrices_from_CIF(CIFpath): 

    xtl = dif.Crystal(CIFpath)
    a, b, c, alpha, beta, gamma = xtl.Cell.lp()    
    Pc2o = xa.Pc2o(a, b, c, alpha, beta, gamma)
    MatSym = xtl.Symmetry.symmetry_matrices 
    MatSymPr = operationsPropresList(MatSym)
    MatSymPrOrtho = []   
    for m in range(len(MatSymPr)):
       mat1 = np.dot(MatSymPr[m], np.linalg.inv(Pc2o))
       EquiOrtho = np.dot(Pc2o, mat1)
       MatSymPrOrtho.append(EquiOrtho)   
    return MatSymPrOrtho


def get_proper_quaternions_from_CIF(CIFpath):
    
    
    #
    # I. Création des opérations de symétrie du groupe ponctuel propre associé
    # au groupe d'espace du cristal, sous forme de quaternions
    #
    
    # ______________________________________________________________________
    # Récupération des opérations de symétrie du groupe d'espace en matrices
    # à l'aide de la librairie Dans_Diffraction (non modifiée car pas d'affichage).
    # Par défaut, cette librairie aligne le repère orthonormé associé au cristal
    # avec b // y, mais ça ne joue pas sur les opérations de symétrie qui sont
    # définies dans le repère du cristal. Donc, pas besoin de modification si on ne
    # utilise que pour récupérer les opérations de symétries du groupe d'espace.
    
    # entrée du fichier .cif du cristal étudié
    # ce fichier doit comporter les opérations de symétrie du groupe d'espace
    

    xtl = dif.Crystal(CIFpath)
    a, b, c, alpha, beta, gamma = xtl.Cell.lp()
    print(xtl.Cell.lp())
    
    # calcul de la matrice passive de passage entre le cristal et le repère orthonormé
    # Cette  matrice prend en entrée un vecteur (ou une position) avec des coordonnées
    # exprimées dans le repère du cristal, et renvoie les coordonnées de ce même vecteur
    # (ou position) dans le repère orthonormé associé au cristal
    # Ce repère est actuellement choisi pour avoir le vecteur a du cristal aligné avec 
    # le vecteur x de la base ortho, et c* du cristal aligné avec z de la base ortho
    
    Pc2o = xa.Pc2o(a, b, c, alpha, beta, gamma)
    # print(Pc2o)
    
    # récupération des matrices de symétrie du cristal, à partir du fichier .cif
    # Attention : si ces matrices (sous la forme x,y,z) ne sont pas présentes dans 
    # le fichier .cif, alors la librairie Dans_Diffraction renvoie une liste avec 
    # seulement une matrice de symétrie (l'identitié)
    # à faire : checker avec le nom du groupe d'espace, et prendre en compte les cas
    # où les opétations de symétrie ne sont pas dans le fichier (travail déjà 
    # effectué par S. Dubail)
    
    MatSym = xtl.Symmetry.symmetry_matrices
    
    # print(xtl.Symmetry.__repr__())
    
    # print(CIFpath)
    # print("les matrices du groupe d'espace correspondant : ")
    # for i in range(len(MatSym)):
    #     print(MatSym[i])
    # print("___________________________________")  
    
    
    # Tri pour isoler toutes les opérations propres du groupe, sans partie translatoire
   
    MatSymPr = operationsPropresList(MatSym)
    MatSymPrOrtho = []
    
    # Pour pouvoir utiliser les quaternions pour les calculs, il faut exprimer les
    # matrices de rotation dans le repère orthonormé associé au cristal, puis les 
    # transformer en quaternions.
    # Explications :
    # Tous les calculs que l'on va faire se passent dans le repère ortho. Donc,
    # un vecteur, une position ou une orientation sont défini dans le repère ortho,
    # puis on fait jouer la symétrie exprimée dans le repère ortho, et le résultat
    # est une orientation équivalente, toujours exprimée dans le repère ortho.
    # Or, la matrice de symétrie est exprimée dans le repère du cristal. Donc, la suite
    # des opérations à faire est la suivante :
    # - passage ortho -> cristal
    # - faire jouer la matrice de symétrie exprimée dans le repère du cristal
    # - revenir dans le repère ortho.
    
    # On a donc (en commençant par la matrice de droite) : 
    # Pc2o - Sym - Po2c
    # Le résultat de cette multiplication est la matrice de symétrie exprimée dans
    # le repère orthonormé, donc utilisable sur des orientations exprimées elles 
    # aussi dans le repère orthonormé
    
    for m in range(len(MatSymPr)):
        mat1 = np.dot(MatSymPr[m], np.linalg.inv(Pc2o))
        EquiOrtho = np.dot(Pc2o, mat1)
        MatSymPrOrtho.append(EquiOrtho)
    
    # print("matrices propres : \n", *MatSymPr, sep='\n')  
    # print("___________________________________")  
    
    # print("matrices propres ortho : \n", *MatSymPrOrtho, sep='\n')  
    # print("___________________________________")  
    
    # transformation des matrices de symétrie // ortho en quaternion
    # vérifié avec MTEX pour les 11 groupes propres
    
    QuatSymPr = []
    Ax = []
    Angle = []
    for i in range(len(MatSymPrOrtho)):
        # print("boucle : ", i, "\n", MatSymPrOrtho[i], "\n")
        # QuatSymPr.append(Quaternion(matrix=MatSymPrOrtho[i]))
        quat = xa.OrientationMatrix2Quat(MatSymPrOrtho[i])
        if quat.scalar < 0.0:
            quat = -quat
        
        QuatSymPr.append(quat)
        Ax.append(QuatSymPr[i].axis)
        Angle.append(QuatSymPr[i].degrees)
    
# La fonction retourne QuatSymPr qui est une liste des quaternions propres correspondant au cristal
# Ces quaternions sont exprimés dans la base ortho liée au cristal

    return QuatSymPr
    
def operationsPropresList(listOpGE):
    # listOpGE est une liste qui contient des tableaux Numpy 3x4 correspondant
    # aux opérations de symétrie. Attention : il y a la partie translatoire aussi !!
    
    OpPropres = []
    # élimination des rotations impropres
    for i in range(len(listOpGE)):
        if np.linalg.det(listOpGE[i][:3, :3]) > 0.0: OpPropres.append(listOpGE[i][:3, :3].flat)
    
    # élimination des doublons    
    tupled_lst = set(map(tuple, OpPropres))
    tupled_lst = list(tupled_lst)
    lst2 = []
    for i in range(len(tupled_lst)):
        lst2.append(np.asarray(tupled_lst[i]).reshape((3, 3)))
    
    return lst2
