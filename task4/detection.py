#detected_points = detect(model, test_img_dir)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from skimage.io import imread
from scipy.misc import imresize
from skimage.util import invert

import numpy as np
import random 
from scipy.misc import imrotate
from math import cos,sin
from keras.preprocessing import image
import glob

from os.path import join
import os

import random 
from scipy.misc import imrotate, imread, imsave, imresize
from math import cos,sin
from keras.preprocessing import image
from os.path import join
import numpy as np

def flip(image, coords):
    def swap(coords, i, j):
        temp = coords[2*i: 2*i + 2].copy()
        coords[2*i: 2*i + 2] = coords[2*j: 2*j + 2].copy()
        coords[2*j: 2*j + 2] = temp
        return
    x_coords = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26])
    image = np.fliplr(np.array(image))
    coords = np.array(coords)
    coords[x_coords] = image.shape[1] - coords[x_coords]
    swap(coords,0,3)
    swap(coords,1,2)
    swap(coords,4,9)
    swap(coords,5,8)
    swap(coords,6,7)
    swap(coords,11,13)
    return image, coords

def rotate(image, coords):
    angle = random.uniform(-13.0,13.0)
    matrix = np.array([[cos(angle*np.pi/180), sin(angle*np.pi/180)] ,
                       [-sin(angle*np.pi/180), cos(angle*np.pi/180)]])
    coords = np.array(coords)
    for i in range(0,14):
        coords[2*i: 2*i+2] = coords[2*i: 2*i+2] - [image.shape[1] / 2., image.shape[0] / 2.]
        coords[2*i: 2*i+2] = matrix.dot(coords[2*i: 2*i+2])
        coords[2*i: 2*i+2] = coords[2*i: 2*i+2] + [image.shape[1] / 2., image.shape[0] / 2.]
    image = imrotate(np.array(image), angle)
    return image, coords

def crop(image, coords):
    image = np.array(image)
    coords = coords.copy()
    from_x = int(image.shape[1]*0.075)
    to_x = image.shape[1] - from_x
    
    from_y = int(image.shape[0]*0.075)
    to_y = image.shape[0] - from_y
    
    x_coords = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26])
    y_coords = np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27])
    
    probability = random.randint(3,5)
    if probability==0:
        image = image[from_y:,from_x:,: ] 
        coords[x_coords] = coords[x_coords] - from_x
        coords[y_coords] = coords[y_coords] - from_y
    elif probability==1:
        image = image[:,from_x:,: ] 
        coords[x_coords] = coords[x_coords] - from_x
    elif probability==2:
        image = image[from_y: ,: ] 
        coords[y_coords] = coords[y_coords] - from_y
    elif probability==3:
        image = image[ :to_y, : ] 
    elif probability==4:
        image = image[ : , :to_x, ] 
    else :
        image = image[:to_y, :to_x, ]
    #print(probability)    
    return image, coords

def data_augmentation_generator(path_features, labels_csv, batch_size, sample_size):
    img_rows, img_cols = 128, 128
    
    filenames = list(labels_csv.keys())[:sample_size]
    
    x_coords = np.array([2 * k for k in range(0, 14)])
    y_coords = np.array([2 * k + 1 for k in range(0, 14)])
    
    while True:
        random.shuffle(filenames)
        for iterator in range(sample_size // batch_size):
            batch_features = np.zeros((batch_size, img_rows, img_cols, 3))
            batch_labels = np.zeros((batch_size, 28))
            
            for i in range(0, batch_size):
                # read picture
                filename = join(path_features, filenames[i + iterator * batch_size])
                pict = imread(filename, mode='RGB')
                coords = labels_csv[filenames[i + iterator * batch_size]].copy()
                
                probability = random.randint(0,1)
                if probability == 1:
                    pict, coords = flip(pict, coords)
                
                probability = random.randint(0,1)
                if probability == 1:
                    pict, coords = rotate(pict, coords)
                
                probability = random.randint(0,1)
                if probability == 1:
                    pict, coords = crop(pict, coords)
                     
                    
                coords[x_coords] = coords[x_coords] / pict.shape[1]
                coords[y_coords] = coords[y_coords] / pict.shape[0] 
                pict = imresize(pict, (img_rows, img_cols), interp = 'bilinear')
                
                batch_features[i] = pict
                batch_labels[i] = coords
            
            batch_features = batch_features.astype('float32') / 255.0
            yield (batch_features, batch_labels)


def train_detector(train_gt, train_img_dir, fast_train):
    
    if fast_train:
        epoch = 1
    else :
        epoch = 100
        
    model = Sequential()


    model.add(Conv2D(16, (3,3),
        input_shape=(128, 128, 3),activation='relu',padding='same', data_format = 'channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3),activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(150,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(28))
    
    optimizer = keras.optimizers.SGD(lr=0.095, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mse',optimizer=optimizer)
    
    count = len(train_gt)

    model.fit_generator(data_augmentation_generator(train_img_dir, train_gt, batch_size=32, sample_size = count*5//6),
                    steps_per_epoch=(count*5//6)//32,
                    epochs=epoch,
                    verbose=1)
                    #validation_data=(data[count*5//6:].astype('float32')/255, y[count*5//6:]/128))
    return

def detect(model, test_img_dir):
    #filenames = glob(join(test_img_dir + '/', '*.jpg'))
    #print(filenames)
    filenames = os.listdir(test_img_dir)
    filenames.sort()
    detected_points = {}
    batch_size = 1000
    img_rows, img_cols = 128,128
    
    x_coords = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26])
    y_coords = np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27])
    
    for j in range(len(filenames)//batch_size):
        batch_features = np.zeros((batch_size, img_rows, img_cols, 3))
        size_images = np.zeros((batch_size , 2))
        for i in range(0,batch_size):
            # read picture
            filename = filenames[i+j*batch_size]
            img = image.load_img(join(test_img_dir,filename))
            pict = image.img_to_array(img)
            size_images[i][0] = pict.shape[0]
            size_images[i][1] = pict.shape[1]
            
            pict = imresize(pict , (img_rows, img_cols,3), interp = 'bilinear')
            batch_features[i] = pict
        batch_labels = model.predict(batch_features / 255.)
        for i in range(0,batch_size):
            coords = batch_labels[i]
            coords[x_coords] = coords[x_coords] * size_images[i][1]
            coords[y_coords] = coords[y_coords] * size_images[i][0]
            
            filename = filenames[i+j*batch_size]
            detected_points[filename] = coords
    
    if (len(filenames)%batch_size!=0):
        size_excess = len(filenames)%batch_size
        k = len(filenames)//batch_size
        batch_features = np.zeros((size_excess, img_rows, img_cols, 3))
        size_images = np.zeros((size_excess , 2))
        for i in range(0,size_excess):
            # read picture
            filename = filenames[i+k*batch_size]
            img = image.load_img(join(test_img_dir,filename))
            pict = image.img_to_array(img)
            size_images[i][0] = pict.shape[0]
            size_images[i][1] = pict.shape[1]

            pict = imresize(pict , (img_rows, img_cols,3), interp = 'bilinear')
            batch_features[i] = pict
        batch_labels = model.predict(batch_features / 255.)
        for i in range(0,size_excess):
            coords = batch_labels[i]
            coords[x_coords] = coords[x_coords] * size_images[i][1]
            coords[y_coords] = coords[y_coords] * size_images[i][0]

            filename = filenames[i+k*batch_size]
            detected_points[filename] = coords
            
    return detected_points