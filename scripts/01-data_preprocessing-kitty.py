# Process raw data and save them into pickle file.
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
# from skimage import io throws an error
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation #image container for loading images and saliency maps
from constants import *

img_size = INPUT_SIZE
salmap_size = INPUT_SIZE



# list of raw saliency maps (including train, test and val)
listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToMaps, '*'))]
# list of all test images in folder of all images
listTestImages = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]

# resizing whole data set:
for currFile in tqdm(listImgFiles): #load images for every available saliency map
    tt = dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'), #imagePath
                                   os.path.join(pathToMaps, currFile + '.jpg'), #saliencyPath
                                   os.path.join(pathToFixationMaps, currFile + '.mat'), #fixationPath 
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                   dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty) #no fixations used

    # if tt.image.getImage().shape[:2] != (480, 640):
    #    print 'Error:', currFile

    imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), img_size, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2BGR)
    saliencyResized = cv2.resize(tt.saliency.getImage(), salmap_size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)
    cv2.imwrite(os.path.join(pathOutputMaps, currFile + '.png'), saliencyResized)



# LOAD DATA

# Training Data
listFilesTrain = [k for k in listImgFiles if 'train' in k]
trainData = []
for currFile in tqdm(listFilesTrain): #all train images from dataset
    trainData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                               os.path.join(pathOutputMaps, currFile + '.png'),
                                               os.path.join(pathToFixationMaps, currFile + '.mat'),
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                               dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty))

with open(os.path.join(pathToPickle, 'trainData.pickle'), 'wb') as f:
    pickle.dump(trainData, f)

# Validation Data
listFilesValidation = [k for k in listImgFiles if 'val' in k]
validationData = []
for currFile in tqdm(listFilesValidation): # changed here: no fixation maps are used
    validationData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                                    os.path.join(pathOutputMaps, currFile + '.png'),
                                                    os.path.join(pathToFixationMaps, currFile + '.mat'),
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                                    dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)) 

with open(os.path.join(pathToPickle, 'validationData.pickle'), 'wb') as f:
    pickle.dump(validationData, f)

# Test Data
testData = []

for currFile in tqdm(listTestImages): # no saliency maps are loaded form the test file, also no fixations.
    testData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                              os.path.join(pathOutputMaps, currFile + '.png'),
                                              os.path.join(pathToFixationMaps, currFile + '.mat'),
                                              dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                              dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty,
                                              dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty))

with open(os.path.join(pathToPickle, 'testData.pickle'), 'wb') as f:
    pickle.dump(testData, f)
