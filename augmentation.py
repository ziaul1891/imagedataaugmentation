import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img ,img_to_array, load_img
import os
import cv2

datagen = ImageDataGenerator(rotation_range =15, 
                     width_shift_range = 0.2, 
                     height_shift_range = 0.2,  
                     rescale=1./255, 
                     shear_range=0.2, 
                     zoom_range=0.2, 
                     horizontal_flip = True, 
                     fill_mode = 'nearest', 
                     data_format='channels_last', 
                     brightness_range=[0.5, 1.5]) 

imgs = os.listdir('G:/Dataset/Corn Disease/data/Blight1')

for img in imgs:
    img=cv2.imread('G:/Dataset/Corn Disease/data/Blight1'+"\\"+img)
    prefix_text="Blight"
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow (x, batch_size=1, save_to_dir =r'G:/Dataset/Corn Disease/data/New', save_prefix ='aug_'+prefix_text, save_format='jpg'):
        i+=1
        if i>10:
            break