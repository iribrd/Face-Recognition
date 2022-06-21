import sqlite3
import numpy as np
import caffe
import os
from os import walk
import shutil
from shutil  import rmtree, move
import io
import faceDetector
import re
class netOptions:
    def __init__ (self):
        self.modelPath='./'
        self.weightPath='./'
        self.netMean=np.array([0, 0, 0])
        self.maxPixelValue=255.0
        self.netDataInputSize=(1,1)
        self.featureLayerName =''                   #Name of th feature layer (generally the last but one layer)


def layerOutput(layerName, net,transformer, img):
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out= net.forward()
    A=net.blobs[layerName].data.copy()
    B=np.reshape(A,(A.shape[1],A.shape[2]*A.shape[3])).flatten()
    return B

def extractFeature (featureLayerName, imgPath, net, transformer): #get an image and return its feature vector
    img = caffe.io.load_image(imgPath)
    featureVec=layerOutput(featureLayerName, net,transformer, img)
    return featureVec

def convertFolderNametoNumeric(parentFolder):
    tempFolder='./temp'
    if os.path.exists(tempFolder):
        shutil.rmtree(tempFolder)
    os.mkdir(tempFolder)
    for __, dirs, __ in os.walk(parentFolder):
        for idx, dirName in enumerate(dirs):
            os.rename(parentFolder + '/' + dirName, tempFolder + '/' + str(idx+1))
    shutil.rmtree(parentFolder, ignore_errors=True)
    shutil.move(tempFolder, parentFolder)
 
def adapt_array(arr):       #Sqlite adapter.
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):    #Sqlite converter.
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)   
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

sqlite3.register_adapter(np.array, adapt_array)    
sqlite3.register_converter("ARRAY", convert_array)


class Gallery:
    def __init__(self, parentFolder, dataBasePath,netOptions):  #netOptions: model path, weight path, mean (3-values)
        self.parentFolder=parentFolder
        self.dataBasePath=dataBasePath
        self.netOptions=netOptions
        #convertFolderNametoNumeric(self.parentFolder)
        border=20
        threshold=[0.3, 0.3, 0.3]
        options=faceDetector.faceDetectionOptions(threshold,border)
        fc=faceDetector.FaceDetector(options)
        fc.cropFolder (self.parentFolder)


    
    def constructGalleryDataBase(self):

        #Load caffe net
        net = caffe.Net(self.netOptions.modelPath, self.netOptions.weightPath, caffe.TEST)

        #Define data layer
        net.blobs['data'].reshape(1,3,self.netOptions.netDataInputSize[0],self.netOptions.netDataInputSize[0])

        #Define Transformer object
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_raw_scale('data', self.netOptions.maxPixelValue)
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_mean('data' , self.netOptions.netMean)
        
        
        
        
        #dataBaseName=os.path.basename(self.dataBasePath)
        #os.path.splitext(dataBaseName)[0]

        if os.path.exists(self.dataBasePath):
            os.remove(self.dataBasePath)
        #sqlite adapter & converter
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("ARRAY", convert_array)

        connection = sqlite3.connect(self.dataBasePath)
        connection.execute('DROP TABLE IF EXISTS persons')
        connection.execute('CREATE TABLE persons (id INTEGER, name text, featureVec ARRAY, PRIMARY KEY (id))')




        
        #dirs=sorted(os.listdir(self.parentFolder),key=char)
        dirs = sorted_aphanumeric(os.listdir(self.parentFolder))
        for dir in dirs:
            files=os.listdir(self.parentFolder +'/' + dir)
            imgsPercase=len(files)
            featureVec=np.zeros(self.netOptions.featureVecLength)
            for imgName in files:
                print('Adding to Gallery Database: %s' %(imgName))
                featureVec=featureVec+extractFeature (self.netOptions.featureLayerName, self.parentFolder +'/' + dir + '/' + imgName, net, transformer)/imgsPercase
                #print(featureVec)
            if files:
                connection.execute('INSERT INTO persons (name,featureVec) values (?,?)' ,(dir,featureVec,))
        connection.commit()
        
        #Read the gallery images
        #for path, subdirs, files in sorted(os.walk(self.parentFolder),key=int):
        #    imgsPercase=len(files)
        #    featureVec=np.zeros(self.netOptions.featureVecLength)
        #    for imgName in files:
        #        print(imgName)
        #        featureVec=featureVec+extractFeature (netOptions.featureLayerName, path + '/' + imgName, net, transformer)/imgsPercase
        #        print(featureVec)
        #    if files:
        #        connection.execute('INSERT INTO persons (featureVec) values (?)' ,(featureVec,))
        #connection.commit()
                


        

       

#parentFolder='E:/public_html/courses/2019_s/Research/offlineRecognition/gallery'
#modelPath='vggface2.prototxt'
#weightPath='vggface2.caffemodel'
#netMean=np.array([91.4953, 103.8827, 131.0912])
#dataBasePath='cases.sqlite'
#maxPixelValue=255.0
#netDataInputSize=(224,224)
#featureLayerName ='pool5/7x7_s1'   
#featureVecLength=2048 
#netOptions=netOptions()
#netOptions.modelPath=modelPath
#netOptions.weightPath=weightPath
#netOptions.netMean=netMean
#netOptions.maxPixelValue=maxPixelValue
#netOptions.netDataInputSize=netDataInputSize
#netOptions.featureLayerName=featureLayerName
#netOptions.featureVecLength=featureVecLength
#gl=Gallery(parentFolder, dataBasePath, netOptions)
#gl.constructGalleryDataBase()








