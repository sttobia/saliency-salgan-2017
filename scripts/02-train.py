#   Two mode of training available:
#       - BCE: CNN training, NOT Adversarial Training here. Only learns the generator network.
#       - SALGAN: Adversarial Training. Updates weights for both Generator and Discriminator.
#   The training used data previously  processed using "01-data_preprocessing.py"
import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne

from tqdm import tqdm
from constants import *
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from utils import *

flag = str(sys.argv[1])


def bce_batch_iterator(model, train_data, validation_sample):
    num_epochs = 301
    n_updates = 1 #flag for number of updates
    nr_batches_train = int(len(train_data) / model.batch_size)
    
    # variables for nice output of error values:
    col1 = 10
    col2 = 30
    str1 = 'Epoch'
    str2 = 'train-loss e'
    header = 'title'
    

       
    pbar_bce = tqdm(total=num_epochs, desc=('TRAINING BCE'))
    
    # print title of the table
    pbar_bce.write('\n\n')
    header_padding = int(round((col1 + col2 - len(header))/2))
    header = ((header_padding * ' ') + header + (header_padding * ' '))
    pbar_bce.write(1*(len(header)*'*' + '\n') + header.upper() + 1*('\n' + len(header)*'*'))
    
    # print header of table
    pbar_bce.write('\n')
    pbar_bce.write(str1.rjust(col1) + str2.rjust(col2))
    pbar_bce.write((len(str1)*'-').rjust(col1) + (len(str2)*'-').rjust(col2))
    
    
    for current_epoch in range(num_epochs):
        
        e_cost = 0.

        random.shuffle(train_data)
        pbar_bce_batch = tqdm(total=nr_batches_train, 
                              desc=('Epoch ' + str(current_epoch) + '/' + str(num_epochs-1)), 
                              leave = False)
        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)

            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            G_cost = model.G_trainFunction(batch_input, batch_output)
            e_cost += G_cost
            n_updates += 1
            pbar_bce_batch.update(1)

        pbar_bce_batch.close()
        e_cost /= nr_batches_train

        # write output:
        pbar_bce.write((str(current_epoch) + '/' + str(num_epochs-1)).rjust(col1) + str(e_cost).rjust(col2))
        
        # Save weights every 5 epochs and predict validation_sample
        if current_epoch % 5 == 0:
            np.savez(DIR_TO_SAVE + 'gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, name = None,
                    path_output_maps=DIR_TO_SAVE)
        pbar_bce.update(1);
    pbar_bce.close()

def salgan_batch_iterator(model, train_data, validation_sample):

    num_epochs = 301
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1 #flag for number of updates
    
    # for creating nice output
    col1 = 10;
    col2 = 30;
    col3 = col2;
    col4 = col2;

    str1 = 'Epoch'
    str2 = 'train-loss g'
    str3 = 'train-loss d'
    str4 = 'train-loss e'
    header = 'training salgan'

    
    pbar_salgan = tqdm(total=num_epochs, desc=('TRAINING SALGAN'))
    
    # print title
    pbar_salgan.write('\n\n')
    header_padding = int(round((col1 + col2 + col3 + col4 - len(header))/2))
    header = ((header_padding * ' ') + header + (header_padding * ' '))
    pbar_salgan.write(1*(len(header)*'*' + '\n') + header.upper() + 1*('\n' + len(header)*'*'))

    # print header
    pbar_salgan.write('\n')
    pbar_salgan.write(str1.rjust(col1) + str2.rjust(col2) + str3.rjust(col3) + str4.rjust(col4))
    pbar_salgan.write((len(str1)*'-').rjust(col1) + (len(str2)*'-').rjust(col2) + (len(str3)*'-').rjust(col3) 
                + (len(str4)*'-').rjust(col4))



    for current_epoch in range(num_epochs):
        g_cost = 0.
        d_cost = 0.
        e_cost = 0.

        random.shuffle(train_data)

        pbar_salgan_batch = tqdm(total=nr_batches_train, 
                                 desc=('Epoch ' + str(current_epoch) + '/' + str(num_epochs-1)),
                                 leave = False)
        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)
            
            # was used for some debugging
            #print ' '
            #print 'Iteration:', n_updates
            #print 'max value salmap: ', 255*np.amax(batch_output)
            #print 'e_cost: ', e_cost, 'd_cost: ', d_cost, 'g_cost: ', g_cost
            #print 'batch_input: ', batch_input 
            #print 'batch output: ', batch_output
            #print ' '

            # train generator with one batch and discriminator with next batch
            if n_updates % 2 == 0:
                G_obj, D_obj, G_cost = model.G_trainFunction(batch_input, batch_output) # call generator train function
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
            else:
                G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output) # call discriminator train function
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost

            n_updates += 1
            pbar_salgan_batch.update(1)

        pbar_salgan_batch.close()
        g_cost /= nr_batches_train
        d_cost /= nr_batches_train
        e_cost /= nr_batches_train
        
        
        # write output
        pbar_salgan.write((str(current_epoch) + '/' + str(num_epochs-1)).rjust(col1) 
            + str(g_cost).rjust(col2) 
            + (str(d_cost)).rjust(col3)
            + (str(e_cost)).rjust(col4))
        
        # Save weights every 3 epoch and predict validation_sample
        if current_epoch % 3 == 0:
            np.savez(DIR_TO_SAVE + 'gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez(DIR_TO_SAVE + 'discrim_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.discriminator['prob']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, name = None, path_output_maps=DIR_TO_SAVE)
        
        pbar_salgan.update(1);
    pbar_salgan.close()
      
            

def train():
    """
    Train both generator and discriminator
    :return:
    """
    # Load data
    print 'Loading training data...'
    # with open('../saliency-2016-lsun/validationSample240x320.pkl', 'rb') as f:
    with open(TRAIN_DATA_DIR, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    # with open('../saliency-2016-lsun/validationSample240x320.pkl', 'rb') as f:
    with open(VALIDATION_DATA_DIR, 'rb') as f:
        validation_data = pickle.load(f)
    print '-->done!'

    # Choose a random sample to monitor the training
    num_random = random.choice(range(len(validation_data)))
    validation_sample = validation_data[num_random]
    print 'Saving random validation sample...'
    cv2.imwrite(DIR_TO_SAVE + '/validationRandomSaliencyGT.png', validation_sample.saliency.data)
    cv2.imwrite(DIR_TO_SAVE + '/validationRandomImage.png', cv2.cvtColor(validation_sample.image.data,
                                                                                cv2.COLOR_RGB2BGR))
    print '-->done!'

    # Create network

    if flag == 'salgan':
        print 'building SalGAN network...'
        model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1])
        print '-->done!\n\n'
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path="nss/gen_", epochtoload=15)
        # load_weights(net=model.discriminator['prob'], path="test_dialted/disrim_", epochtoload=54)
        salgan_batch_iterator(model, train_data, validation_sample.image.data)

    elif flag == 'bce':
        print 'building BCE network...'
        model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1])
        print '-->done!\n\n'
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path='test/gen_', epochtoload=15)
        bce_batch_iterator(model, train_data, validation_sample.image.data)
    else:
        print "Invalid input argument."
if __name__ == "__main__":
    train()
