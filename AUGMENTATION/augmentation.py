# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:08:52 2019

@author: Lenovo
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage import exposure, color
from skimage.transform import resize
from keras import backend as K
import glob
from keras.preprocessing.image import ImageDataGenerator


def imgGen(img, zca=False, rotation=0., w_shift=0., h_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False,  preprocess_fcn=None, batch_size=9):
    datagen = ImageDataGenerator(
            zca_whitening=zca,
            rotation_range=rotation,
            width_shift_range=w_shift,
            height_shift_range=h_shift,
            shear_range=shear,
            zoom_range=zoom,
            fill_mode='nearest',
            horizontal_flip=h_flip,
            vertical_flip=v_flip,
            preprocessing_function=preprocess_fcn,
            data_format=K.image_data_format())
    
    datagen.fit(img)
    
    i=0
    j=0
    for img_batch in datagen.flow(img, batch_size=9, shuffle=False):
        
        for img in img_batch:
            plt.subplot(330 + 1 + i)
            plt.imshow(img)
            
            for v in img_batch:
                filename =r"F:/S.K/AUGMENTATION/water_body_Aug/water_"+str(j)+".jpg"
                imsave(filename,v )
                j+=1
            i=i+1  
        
        if i >= batch_size:
            break
      
 
    plt.show()
# Contrast stretching
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

# Histogram equalization
def HE(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

# Adaptive histogram equalization
def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq

    
for img in glob.glob("F:\S.K\AUGMENTATION\water_body\*.jpg"):
    img = imread(img)
    #plt.imshow(img)
    #plt.show()
    
    # reshape it to prepare for data generator
    img = img.astype('float32')
    img /= 255
    h_dim = np.shape(img)[0]
    w_dim = np.shape(img)[1]
    num_channel = np.shape(img)[2]
    img = img.reshape(1, h_dim, w_dim, num_channel)
    
    # print(img.shape)
   
    # generate images using function imgGen
    imgGen(img, rotation=30, h_shift=0.5, preprocess_fcn = contrast_stretching)
   # imgGen(img, rotation=30, h_shift=0.5, preprocess_fcn = HE)
    #imgGen(img, rotation=30, h_shift=0.5, preprocess_fcn = AHE)