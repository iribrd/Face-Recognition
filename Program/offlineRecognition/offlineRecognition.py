import cv2
import faceDetector
import Gallery
import caffe
from scipy import spatial
import numpy as np
import Gallery
import sqlite3
import io
import os
import shutil
import subprocess
#caffe.set_mode_gpu()
#caffe.set_device(0)

def layerOutput(layerName, net,transformer, img):
    #img2=transformer.preprocess('data', img)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out= net.forward()
    A=net.blobs[layerName].data.copy()
    #B=np.reshape(A,(A.shape[1],A.shape[2]*A.shape[3])).flatten()
    B=A.flatten()

    return B
 
def calculateDistanceArray (galleryFeatureVecs,testFeatureVec,distanceThr):  
    distance=np.zeros(galleryFeatureVecs.shape[0])
    for i in range(galleryFeatureVecs.shape[0]):
        C=galleryFeatureVecs[i,0,:]
        distance[i] =spatial.distance.cosine(testFeatureVec, C) # spatial.distance.cosine(testFeatureVec, C)  #  spatial.distance.cosine(testFeatureVec, C) 
        #distance2 =spatial.distance.correlation (testFeatureVec,C)
        #distance[i]=distance1#(distance1+distance2)/2
    return distance

def adapt_array(arr):       #Sqlite adapter.
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):    #Sqlite converter.
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)   

def constructGalleryDataBase(GalleryParentFolder,netOptions, dataBasePath ): #Create gallery database 
    gl=Gallery.Gallery(GalleryParentFolder, dataBasePath, netOptions)
    gl.constructGalleryDataBase()
sqlite3.register_adapter(np.array, adapt_array)    
sqlite3.register_converter("ARRAY", convert_array)
#===================================================================================================
#Read critical Settings (Must be set in config.txt):
configTxtFile=open('config.txt', 'r')
#-------------------------------------------------------------------------------------------------------
GalleryParentFolder=configTxtFile.readline().split('@')[1].strip()[1:-1]
videoFilesFolder=configTxtFile.readline().split('@')[1].strip()[1:-1]
newGallery=configTxtFile.readline().split('@')[1].strip()
if newGallery == 'True':
    newGallery=True
else:
    newGallery=False
distanceThr=float(configTxtFile.readline().split('@')[1].strip())
frameStep=int(configTxtFile.readline().split('@')[1].strip())
#===================================================================================================



#===================================================================================================
#Programmer settings (should be set by the programmer. Generally user does not need to change them).
#===================================================================================================
#---------------------------------------------------------------------------------------------------
#Deep neural network settings
netOptions=Gallery.netOptions()
netOptions.modelPath='vggface2.prototxt'
netOptions.weightPath='vggface2.caffemodel'
netOptions.netMean=np.array([91.4953, 103.8827, 131.0912])
netOptions.maxPixelValue=255.0
netOptions.netDataInputSize=(224,224)
netOptions.featureLayerName='pool5/7x7_s1'   
netOptions.featureVecLength=2048 
#---------------------------------------------------------------------------------------------------
#Gallery database settings
dataBasePath='cases.sqlite'
#---------------------------------------------------------------------------------------------------
#Log file settings
#logFilePath = open('log.txt', 'w')
#---------------------------------------------------------------------------------------------------
#Face detector settings
cropBorder=0
faceThresholds=[0.7, 0.7, 0.7]

#distanceThr=0.35
#Folder for the algorithm results
foundFolderParent='./found'
notFoundFolder='./notFound'
##Resize values
#imgResizeDim=(1280,720)  #(w,h)
#===================================================================================================


#=====================================================================================================================
#                                           Main body
subprocess.call('nircmd.exe win close class "CabinetWClass"' , shell=True)
#Construct gallery database based on images in GalleryParentFolder (Image for each person must be in one child folder)
if newGallery == True:
    constructGalleryDataBase(GalleryParentFolder,netOptions, dataBasePath )

#Construct face Detection object
faceOptions=faceDetector.faceDetectionOptions(faceThresholds,cropBorder)
fc=faceDetector.FaceDetector(faceOptions)



#Initialize caffe-based DNN and transformer
net = caffe.Net(netOptions.modelPath, netOptions.weightPath, caffe.TEST)
net.blobs['data'].reshape(1,3,netOptions.netDataInputSize[0],netOptions.netDataInputSize[0])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', netOptions.maxPixelValue)
#transformer.set_channel_swap('data', (2,1,0))
transformer.set_mean('data' , netOptions.netMean)

#Initialize found Folder (delete if exists and create a new one)
if os.path.exists(foundFolderParent):
    shutil.rmtree(foundFolderParent, ignore_errors=True)

os.mkdir(foundFolderParent)


if os.path.exists(notFoundFolder):
    shutil.rmtree(notFoundFolder, ignore_errors=True)

os.mkdir(notFoundFolder)

#Load galerry feture vectors to RAM
cursor = sqlite3.connect(dataBasePath,detect_types=sqlite3.PARSE_DECLTYPES).cursor()
cursor.execute('SELECT featureVec FROM persons')
galleryFeatureVecs = np.array(cursor.fetchall())
#cursor.close()
connection = sqlite3.connect(dataBasePath)
connection.execute('DROP TABLE IF EXISTS finds')
connection.execute('CREATE TABLE finds  (id INTEGER PRIMARY KEY AUTOINCREMENT,galleryID INTEGER, videoID INTEGER,second INT, frame INT, fileName TEXT )')
connection.execute('DROP TABLE IF EXISTS finds2')
connection.execute('CREATE TABLE finds2  (id INTEGER PRIMARY KEY AUTOINCREMENT, videoID INTEGER,galleryID INTEGER,result TEXT)')
connection.execute('DROP TABLE IF EXISTS videos')
connection.execute('CREATE TABLE videos  (videoID INTEGER, name TEXT, fps INTEGER, description TEXT, PRIMARY KEY (videoID))')

connection.execute('CREATE TABLE IF NOT EXISTS params  (found_path_root TEXT, notfound_path_root TEXT)')
connection.execute('INSERT INTO params(found_path_root,notfound_path_root) VALUES (?,?)' ,(os.path.abspath(foundFolderParent),os.path.abspath(notFoundFolder)))

videoFilePaths=[(videoFilesFolder + '/' + videoFileName) for videoFileName in os.listdir(videoFilesFolder)]

recongitedFile = open('recongitedFile.txt','w')

cursor.execute('select COUNT(*) from persons') 
size_of_galary = cursor.fetchone()[0]
vID = 0
for videoFileName in os.listdir(videoFilesFolder):
    Foundlist = []
    for i in range(size_of_galary+1): 
        Foundlist.append([])
    vID = vID + 1
    recongitedFile.write('#\n')
    videoFilePath = videoFilesFolder + '/' + videoFileName
    #connection.commit()
    #videoFileName=os.path.splitext(os.path.basename(videoFilePath))[0]
    #Read video 
    cap = cv2.VideoCapture(videoFilePath)
    fps=cap.get(cv2.CAP_PROP_FPS);
    connection.execute('INSERT INTO videos(videoID,name,fps) VALUES (?,?,?)' ,(vID,videoFileName,fps,))
    frameNums=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cnt=-1
    while cap.isOpened():
    
        for j in range(frameStep):
            cnt=cnt+1
            ret, frame = cap.read()
        print('File: %s - Frame %d Processed....'%(videoFileName,cnt))
        if  frame is not None:
            colStart, colStop, rowStart, rowStop=fc.cropFace (frame )
            for i in range(colStart.shape[0]):
                img=frame[rowStart[i]:rowStop[i],colStart[i]:colStop[i]]/255.
                #img = cv2.resize(img, imgResizeDim) # default is bilinear
                #img = img / 255.
                testFeatureVec=layerOutput(netOptions.featureLayerName, net,transformer, img)
                #print(testFeatureVec)
                distance=calculateDistanceArray (galleryFeatureVecs,testFeatureVec,distanceThr)
                minDist=np.min(distance)
                minDistIdx=np.argmin(distance)+1
                if minDist <= distanceThr:
                    #logFilePath.write('Found Person %d : (File:%s     )(Time:  %d  second)\n' %(minDistIdx, videoFilePath, int(cnt/fps)))
                    if  not os.path.exists(foundFolderParent + '/' + str(minDistIdx)):
                        os.mkdir(foundFolderParent + '/' + str(minDistIdx))
                    foundedFacePath = foundFolderParent + '/' + str(minDistIdx) + '/' + videoFileName +'_'+ str(int(cnt)) + '_'+str(int(cnt/fps)) +'_id ' + str(minDistIdx) + '.jpg'
                    #recongitedFile.write()
                    cv2.imwrite(foundedFacePath , img*255)
                    #cursor.execute('SELECT videoID FROM videos WHERE name=?',(videoFileName,))
                    #vID = np.array(cursor.fetchone())[0]
                    #print (os.path.abspath( foundedFacePath ))
                    #connection.execute('INSERT finds (galleryID,videoID,second, frame, pic) values (?,?,?,?,?)' ,(minDistIdx,vID, int(cnt/fps),int(cnt),os.path.abspath( foundedFacePath )))  
                    Foundlist[minDistIdx].append( str(int(cnt)) )
                    connection.execute('INSERT INTO finds (galleryID,videoID,second, frame, fileName) values (?,?,?,?,?)' ,(minDistIdx,vID, int(cnt/fps),int(cnt), foundedFacePath))            
                    connection.commit()
                    #cv2.imwrite(foundFolderParent + '/' + str(minDistIdx) + '/'  + videoFileName + '_Frame_time_' + str(int(cnt/fps)) + '_personID_' + str(minDistIdx) + '.jpg' , frame)
                else:
                    cv2.imwrite(notFoundFolder + '/' + videoFileName +'_f' + str(cnt) + '_s' + str(int(cnt/fps)) + '_' + str(i) + '.jpg' , img*255)
                    #cv2.imwrite(notFoundFolder + '/' + videoFileName + '_Frame_time_' + str(int(cnt/fps)) + '.jpg' , frame)
                #connection.commit()
                    
        else:
            cap.release()
    for i in range(size_of_galary+1):
        res = '' 
        if len(Foundlist[i]) != 0:
            for x in Foundlist[i]:
                res += x + ','
            connection.execute('INSERT INTO finds2 (videoID,galleryID,result) values (?,?,?)' ,(vID,i,res ))        
#logFilePath.close()

