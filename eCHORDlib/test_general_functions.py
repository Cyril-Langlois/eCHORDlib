# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:31:02 2023

@author: clanglois1
"""
import math
import os
import h5py
import numpy as np
from LibrairiesCyril import Fct_profil_modification as fct
from LibrairiesCyril import general_functions as gf


_, _, fileP, keys = gf.openH5file()

print(keys)
print()
print(fileP)
print()

# print(dirFile)