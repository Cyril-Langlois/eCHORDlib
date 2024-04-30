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

from eCHORDlib import general_functions as gf


path2thisFile = abspath(getsourcefile(lambda:0))
uiclass, baseclass = pg.Qt.loadUiType(os.path.dirname(path2thisFile) + "/stackViewer_v2.ui")

class MainWindow(uiclass, baseclass):
    # def __init__(self, rawImage, remOutImage, parent):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.expStack = parent.image
        self.expStack = np.flip(self.expStack, 1)
        self.expStack = np.rot90(self.expStack, k=1, axes=(2, 1))

        
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        self.expSeries.setImage(self.expStack)  
        self.expSeries.autoRange()

        self.x = 0
        self.y = 0
        
           
        self.crosshair_v1= pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h1 = pg.InfiniteLine(angle=0, movable=False)

        self.expSeries.addItem(self.crosshair_v1, ignoreBounds=True)
        self.expSeries.addItem(self.crosshair_h1, ignoreBounds=True) 
        
        self.plotIt = self.profiles.getPlotItem()
        self.plotIt.addLine(x = self.expSeries.currentIndex)
        
        self.proxy1 = pg.SignalProxy(self.expSeries.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy4 = pg.SignalProxy(self.expSeries.ui.graphicsView.scene().sigMouseClicked, rateLimit=60, slot=self.mouseClick)
        
        
        self.expSeries.timeLine.sigPositionChanged.connect(self.drawCHORDprofiles)


    def displayExpStack(self, Series):
        # self.expSeries.clear()
        self.expSeries.ui.histogram.hide()
        self.expSeries.ui.roiBtn.hide()
        self.expSeries.ui.menuBtn.hide()
        self.expSeries.setImage(Series)  
        self.expSeries.autoRange()
    
    def drawCHORDprofiles(self):
        try:
            self.profiles.clear()
            self.plotIt.addLine(x = self.expSeries.currentIndex)

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
            self.lineEdit.setText(str(self.x) + " ; " + str(self.y))
       
class stackView(object):
    def __init__(self, image):
 
        self.image = gf.convertToUint8(image)
        self.stackViewLaunch()
        
    def stackViewLaunch(self):
        app = QApplication(sys.argv)
        w = MainWindow(self)
        w.show()
        app.setQuitOnLastWindowClosed(True)
        app.exec_()
    

if __name__ == '__main__':
    
    StackLoc, StackDir = gf.getFilePathDialog("Stack d'image' (*.tiff)")
    
    Raw_Stack = tf.TiffFile(StackLoc[0]).asarray()
    b = stackView(Raw_Stack)





