# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:26:20 2023

@author: clanglois1
"""
import os
import sys

from inspect import getsourcefile
from os.path import abspath

import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import tifffile as tf

from LibrairiesCyril import general_functions as gf

path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/remOutliers_v3.ui")

class MainWindow(uiclass, baseclass):

    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        self.expStack = parent.image
        self.expStack = np.flip(self.expStack, 1)
        self.expStack = np.rot90(self.expStack, k=1, axes=(2, 1))

        self.denoised_Stack = np.copy(parent.remOutImage)
        self.denoised_Stack = np.flip(self.denoised_Stack, 1)
        self.denoised_Stack = np.rot90(self.denoised_Stack, k=1, axes=(2, 1))
        
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        self.expSeries.setImage(self.expStack)  
        self.expSeries.autoRange()

        self.x = 0
        self.y = 0
        
        self.radius = 2
        self.threshold = 10
        
        self.label_radius.setText("Radius : " + str(self.radius))
        self.label_threshold.setText("Threshold : " + str(self.threshold))
        
        if self.denoised.isChecked():
            self.denoised.toggle()
            
        self.denoised.setCheckable(False)

        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False)

        self.expSeries.addItem(self.crosshair_v1, ignoreBounds=True)
        self.expSeries.addItem(self.crosshair_h1, ignoreBounds=True) 
        
        self.plotIt = self.profiles.getPlotItem()
        self.plotIt.addLine(x = self.expSeries.currentIndex)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        # self.slider_threshold.sliderReleased.connect(self.radius_changed)
        
        self.slider_threshold.valueChanged.connect(self.threshold_changed)
        self.slider_radius.valueChanged.connect(self.radius_changed)
        self.expSeries.timeLine.sigPositionChanged.connect(self.drawCHORDprofiles)
        self.preview.clicked.connect(self.remOutStack)
        self.denoised.stateChanged.connect(self.drawCHORDprofiles)

    def displayExpStack(self, Series):
        # self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        
        view = self.expSeries.getView()
        state = view.getState()        
        self.expSeries.setImage(Series) 
        view.setState(state)
    
    def radius_changed(self):
        self.radius = self.slider_radius.value()
        print("radius changed : ", self.radius, "  threshold : ", self.threshold)
        self.label_radius.setText("Radius : " + str(self.radius))
        self.remOutSlice()
    
    def threshold_changed(self):
        self.threshold = self.slider_threshold.value()
        print("threshold changed : ", self.threshold, "  radius : ", self.radius)
        self.label_threshold.setText("Threshold : " + str(self.threshold))
        self.remOutSlice()
    
    def remOutSlice(self):
        print("remOutSlice entered")
        
        dummy, a = gf.remove_outliers(self.expStack[0, :, :], self.radius, self.threshold)
        
        self.denoised_Stack[0, :, :] = a 
        
        self.displayExpStack(self.denoised_Stack)
        
        self.Modified_pixels_proportion, self.mean_modified_pixels_intensity, self.mean_proportion_intensity = gf.Pixels_transformation(self.expStack[0, :, :], self.denoised_Stack[0, :, :])
        print("Taux de pxls modifié (%) : ", '%.1f'%(self.Modified_pixels_proportion), "  taux de transformation (%) : ", '%.2f'%(self.mean_proportion_intensity))
        
        self.denoised.setEnabled(False)

    def remOutStack(self):
        print("remOutStack entered")
        self.denoised.setEnabled(False)
        self.preview.setEnabled(False)
        
        for i in range(0,len(self.expStack[:, 0, 0])): # Applique pour chaque slice les paramètres du remove outlier
            _, self.denoised_Stack[i, :, :] =  gf.remove_outliers(self.expStack[i,:,:], self.radius, self.threshold)
            if (i+1)%10 == 0:
                print(f"slice {i+1} denoised")
        print("removing outliers achieved")        
        self.drawCHORDprofiles()                

        self.denoised.setEnabled(True)         
        self.denoised.setCheckable(True)
        self.denoised.setChecked(True)
        self.preview.setEnabled(True)
        
        self.parent.remOutImage = np.copy(self.denoised_Stack)
        self.parent.remOutImage = np.rot90(self.parent.remOutImage, k=3, axes=(2, 1))
        self.parent.remOutImage = np.flip(self.parent.remOutImage, 1)
        
    def drawCHORDprofiles(self):
        try:
            self.profiles.clear()
            self.plotIt.addLine(x = self.expSeries.currentIndex)
            # self.plotIt.addLine(x = 120)
            if self.denoised.isChecked():
                self.profiles.plot(self.denoised_Stack[:, self.x, self.y], pen=(255, 255, 23))
            self.profiles.plot(self.expStack[:, self.x, self.y], pen=(255, 125, 200))
        except:
            pass

    def mouseMoved(self, e):

        pos = e[0]

        # print("mouseMoved", pos)
        if not self.mouseLock.isChecked():
            if self.expSeries.view.sceneBoundingRect().contains(pos):
    
                item = self.expSeries.view
                mousePoint = item.mapSceneToView(pos) 
                     
                self.crosshair_v1.setPos(mousePoint.x())
                self.crosshair_h1.setPos(mousePoint.y())

            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
            # print(self.x, self.y)
            if self.x >= 0 and self.y >= 0 and self.x < len(self.expStack[0, :, 0]) and self.y < len(self.expStack[0, 0, :]):
                self.drawCHORDprofiles()

    
    def mouseClick(self, e):

        pos = e[0]
        
        self.mouseLock.toggle()
        
        fromPosX = pos.scenePos()[0]
        fromPosY = pos.scenePos()[1]
        
        posQpoint = QtCore.QPointF()
        posQpoint.setX(fromPosX)
        posQpoint.setY(fromPosY)

        if self.expSeries.view.sceneBoundingRect().contains(posQpoint):
                
            item = self.expSeries.view
            mousePoint = item.mapSceneToView(posQpoint) 

            self.crosshair_v1.setPos(mousePoint.x())
            self.crosshair_h1.setPos(mousePoint.y())
                 
            self.x = int(mousePoint.x())
            self.y = int(mousePoint.y())
            
        # print(self.x, self.y)
        if self.x >= 0 and self.y >= 0 and self.x < len(self.expStack[0, :, 0])and self.y < len(self.expStack[0, 0, :]):
            self.drawCHORDprofiles()
       
class launcher(object):
    def __init__(self, image, remOutImage):
      
        self.image = image
        self.remOutImage = remOutImage 
       
        self.remOutGUI()
        
    def remOutGUI(self):
        app = QApplication(sys.argv)
        w = MainWindow(self)
        w.show()
        app.setQuitOnLastWindowClosed(True)
        app.exec_()
    

if __name__ == '__main__':
    
    StackLoc, StackDir = gf.getFilePathDialog("Stack d'image' (*.tiff)")
    
    Raw_Stack = tf.TiffFile(StackLoc[0]).asarray()
    b = launcher(Raw_Stack, np.copy(Raw_Stack))





