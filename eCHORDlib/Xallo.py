# -*- coding: utf-8 -*-
import numpy as np
from pyquaternion import Quaternion

def MetricTensor(a, b, c, alpha, beta, gamma): 
    """
    entrée des longueurs en Angtröms, entrée des angles en degrés
    
    Parameters:
        a (float): angle alpha
        b (float): angle beta
    
    """

    metricTensor = np.matrix(
    [[a * a, a * b * cos(toRad(gamma)), a * c * cos(toRad(beta))],
    [a * b * cos(toRad(gamma)), b * b, b * c * cos(toRad(alpha))],
    [a * c * cos(toRad(beta)), b * c * cos(toRad(alpha)), c * c]])

    # exemple d'utilisation :
    # Calcul d'un produit scalaire dans un cristal non orthonormée entre V1 et V2 :
    # V1.V2 = (V1.T * M * V2) avec .T la transposée du vecteur colonne, qui est
    # donc un vecteur horizontal
    
    return metricTensor


def Pc2o(a, b, c, alpha, beta, gamma): 
    # entrée des longueurs en Angtröms, entrée des angles en degrés
    # Retourne la matrice qui permet d'avoir les coordonnées d'un vecteur du cristal (c)
    # dans le repère orthonormé associé (o)
    # Le repère orthonorme est ici considere comme x // a ; z // c* et y // z*x
	# Le résultat est donc une matrice passive.
    
    Volume = (np.linalg.det(MetricTensor(a, b, c, alpha, beta, gamma)))**0.5
    # print("Volume de la maille : ", Volume)
    paramF = cos(toRad(beta)) * cos(toRad(gamma)) - cos(toRad(alpha))
    
    matrix = np.matrix(
    [[a, b * cos(toRad(gamma)), c * cos(toRad(beta))],
    [0.0, b * sin(toRad(gamma)), - c * paramF / sin(toRad(gamma))],
    [0.0, 0.0, Volume / a / b / sin(toRad(gamma))]])
    
    return matrix

def cos(a):
	# entrer l'angle en radians
    b = np.cos(a)
    return b

def sin(a):
	# entrer l'angle en radians
    b = np.sin(a)
    return b

def toRad(e):
    f = float(e)/360.0*2.0*np.pi
    return f

def toDeg(e):
    f = float(e)/2.0/np.pi*360.0
    return f

def OrientationMatrix2Quat(om):
    quat = Quaternion(matrix = om.T)
    if quat.scalar < 0.0: quat = -quat
    return quat

def QuaternionToAxisAngle(quat):
    quat1 = Quaternion(quat)
    return quat1.axis, quat1.degrees

def disOfromQuatSymNoMat(quat1, quat2, listSymQ):
    """Disorientation calculation.
    
    This function takes as input two quaternions representing orientations of two crystals
    OF SAME POINT GROUP (same set of symetry operations) and return the axis and angle 
    representing the disorientation between them, with the smallest angle considering
    the symetries of the two crystals.
    
    Keyword Arguments:
    -----------------
    quat1 -- Quaternion representing the orientation of the first crystal. Type : Quaternion object from library pyquaternion
    quat2 -- Quaternion representing the orientation of the second crystal. Type : Quaternion object from library pyquaternion
    listSymQ -- list of symetry operations expressed as quaternions. Type : python list containing Quaternion object from library pyquaternion
    
    Returns :
    axis : axis of disorientation. Type : 1D Numpy array with three components
    angle : disorientation in degrees. Type : float
    """
    EquivalentO = Quaternion(array = np.zeros(4))
    resultO = Quaternion(array = np.zeros(4))
    
    if  type(quat1) != type(resultO):
        quat1 = Quaternion(quat1)
        quat2 = Quaternion(quat2)
    
    
    disOrientation = 7000.0

    ori = quat2.inverse * quat1

    for m in listSymQ:
        
        EquivalentO =  m * ori             # que l'on prenne m ou m.inverse
                                           # n'a pas d'importance car les deux figurent dans listSymM
                                           
        # la valeur absolue prend le rôle de la switching symetry entre A -> B ou B -> A
        misOrientation =  abs(EquivalentO.degrees) 
        axis = EquivalentO.axis
        
        # u = axis[0]
        # v = axis[1]
        # w = axis[2]
        
        # if (((0.0 <= w) and (w<=v) and (v <= u)) or ((0.0 <= -w) and (-w<=-v) and (-v <= -u))):
        #     if misOrientation <= disOrientation: 
        #         resultO = EquivalentO
        #         disOrientation = misOrientation

        if misOrientation <= disOrientation: 
            resultO = EquivalentO
            disOrientation = misOrientation
            # print(resultO.degrees, resultO.axis)
                
    omega = resultO.degrees
    axis = resultO.axis

    omega = abs(omega)

    return axis, omega