import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print os.path.join(path_to_images, curr_file + '.png')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, num_epoch=None, name=curr_file, path_output_maps=path_output_maps)


def main():
    # Create network, for predicting only BCE part is used.
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model snapshot
    load_weights(model.net['output'], path='weights/own/gen_', epochtoload=90)
    # Here need to specify the path to images and output path
    test(path_to_images='../images/', path_output_maps='../saliency/', model_to_test=model)
    
    # CHANGE HERE: USE ABSOLUT OUTPUT PATH

if __name__ == "__main__":
    main()
