import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
from eliaLib import dataRepresentation #image container for loading images and saliency maps

# specify if testing whole dataset or not
train_data = 1
# specifiy dataset to load weights from
dataset_name_weights = '50_25_25_salgan_v2'
dataset_name_images = '50_25_25'
# specify epoch to load weights from:
epoch_no = 300

def test(path_to_images, path_output_maps, model_to_test=None):
    # Load Data
    if train_data:
        # set up different input and output directory:
        path_to_images = (HOME_DIR + 'dataset/' + dataset_name_images + '/raw_data/images/' )
        path_output_maps= (HOME_DIR + 'prediction/')
        
        # get filenames:
        listTestFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*test*'))]
        
        # preparing training data sets:
        for currFile in tqdm(listTestFiles, desc = 'predicting dataset'): #load test images
            tt = dataRepresentation.Target(os.path.join(path_to_images, currFile + '.jpg'), #imagePath
                                           os.path.join(pathToMaps, currFile + '.jpg'), #saliencyPath
                                           os.path.join(pathToFixationMaps, currFile + '.mat'), #fixationPath 
                                           dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                           dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty,
                                           dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)
      
    
            curr_img = cv2.cvtColor(tt.image.getImage(),cv2.COLOR_RGB2BGR)
            predict(model=model_to_test, image_stimuli=curr_img, 
                    num_epoch=None, name=currFile, path_output_maps=path_output_maps)
        
    else:
        list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]

        list_img_files.sort()
        for curr_file in tqdm(list_img_files, desc = 'predicting images', ncols=20):
            print os.path.join(path_to_images, curr_file + '.png')
            img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            predict(model=model_to_test, image_stimuli=img, num_epoch=None, name=curr_file, path_output_maps=path_output_maps)


def main():
    # Create network, for predicting only BCE part is used.
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model snapshot
    load_weights(model.net['output'], path=('output/' + dataset_name_weights + '/gen_'), epochtoload=epoch_no)
    # Here need to specify the path to images and output path
    test(path_to_images='../images/', path_output_maps='../saliency/', model_to_test=model)
    

if __name__ == "__main__":
    main()
