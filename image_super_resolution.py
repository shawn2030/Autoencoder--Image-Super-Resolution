# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:47:48 2020

@author: shoun
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import add
from keras.layers import UpSampling2D
from keras import regularizers

############     AUTOENCODER    ##############


###########  BUILDING THE ENCODER  ###############

input_img = Input(shape = (256,256,3))      #made a tensor of shape (256,256,3)

l1 = Conv2D(64 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(input_img)
l2 = Conv2D(64 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l1)

l3 = MaxPooling2D(pool_size = (2,2) , padding = 'same')(l2)
l3 = Dropout(0.3)(l3)

l4 = Conv2D(128 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l3)
l5 = Conv2D(128 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l4)

l6 = MaxPooling2D(pool_size = (2,2) , padding = 'same')(l5)

l7 = Conv2D(256, (3,3), padding = 'same', activation = 'relu', activity_regularizer = regularizers.l1(10e-10))(l6)

encoder = Model(input_img,l7)
encoder.summary()


############    BUILDING THE DECODER    ################

l8 = UpSampling2D()(l7)     #opposite of maxpooling

l9 = Conv2D(128 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l8)
l10 = Conv2D(128 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l9)

l11 = add([l5 , l10])       #MERGE or using residual/skip connections

l12 = UpSampling2D()(l11)

l13 = Conv2D(64 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l12)
l14 = Conv2D(64 , (3,3) , padding = 'same' , activation = 'relu' , activity_regularizer = regularizers.l1(10e-10))(l13)

l15 = add([l14 , l2])       #MERGE or using residula/skip connections

#channel = 3 for RGB since it is the original input that was RGB
decoded = Conv2D(3, (3,3), padding='same', activation = 'relu' ,activity_regularizer= regularizers.l1(10e-10))(l15)

###############     BUILDING THE AUTO_ENCODER   ##################
autoencoder = Model(input_img , decoded)

autoencoder.summary()

autoencoder.compile(optimizer = 'adadelta' , loss = 'mean_squared_error')


#############   CREATING THE REQUIRED DATASET   ##################

import os
import re
from skimage.transform import resize,rescale
import  matplotlib.pyplot as plt
import numpy as np

def train_batches(just_load_dataset = False):
    
    batches = 16   #number of images to have at the same time in a batch
    
    batch = 0       #number of images in a current batch (increases per loop and then resets for each batch)
    batch_nb = 0    #current batch index is stored
    
    max_batches = -1    #If you want to train only limited number of images to finish the training even faster.
    
    epoch = 5
    
    images = []
    x_train_n = []
    x_train_down = []
    
    x_train_n2 = []     #resulting high resolution dataset
    x_train_down2 = []  #resulting low resolution dataset
    
    for root , dirnames , filenames in os.walk("D:/Kaggle and Projects/Image Super Resolution/2Pkn2dIvSzu5J9nSL_s77w_d428a2188ff44626be89908f348634e6_Completed_Notebook_Data_Autoencoders/Completed_Notebook_Data_Autoencoders/data/rez/cars_train/"):
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                
                if batch_nb == 32:
                    return x_train_n2 , x_train_down2
                
                filepath = os.path.join(root,filename)
                image = plt.imread(filepath)
                
                if len(image.shape) > 2:
                    
                    image_resized = resize(image , (256 , 256))     #Resizing every image running through the loop to be of same size when givena s input to autoencoder
                    x_train_n.append(image_resized)                 #Adding this resized image to HIGH RES dataset
                    x_train_down.append(rescale(rescale(image_resized , 0.5), 2.0))     #rescaling it 0.5x and then again by 2.0x to get lower resolution image but still reamins 256 x 256
                    
                    batch = batch + 1
                    
                    if batch == batches:
                        batch_nb = batch_nb +1
                        
                        x_train_n2 = np.array(x_train_n)
                        x_train_down2 = np.array(x_train_down)
                        
                        if just_load_dataset:
                            return x_train_n2 , x_train_down2
                        
                        print('Training Batch', batch_nb, '(' , batches,' )' )
                        
                        autoencoder.fit(x_train_down2 , x_train_n2,
                                        epochs = epoch,
                                        batch_size = 8,
                                        shuffle = True,
                                        validation_split = 0.2)
                        
                        x_train_n = []
                        x_train_down = []
                        
                        batch = 0
                        
    return x_train_n2 , x_train_down2


x_train_n , x_train_down = train_batches(just_load_dataset= False)

encoded_imgs = encoder.predict(x_train_down)

# We clip the output so that it doesn't produce weird colors
sr1 = np.clip(autoencoder.predict(x_train_down), 0.0, 1.0)

image_index = 16



plt.figure(figsize=(64, 64))
i = 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_down[image_index], interpolation="bicubic")
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(encoded_imgs[image_index].reshape((64*64, 256)))
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(sr1[image_index])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_n[image_index])
plt.show()