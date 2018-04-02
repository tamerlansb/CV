from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

from skimage.io import imread
from skimage.util import invert

import numpy as np
import random 
from math import cos,sin
from keras.preprocessing import image
import glob

from os.path import join
import os

import random 
from scipy.misc import imrotate, imread, imsave, imresize
from math import cos,sin
from keras.preprocessing import image
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD


def flip(image, y):
    image = np.fliplr(image)
    return image, y

def rotate(image, y):
    angle = random.uniform(-13.0,13.0)
    image = imrotate(image, angle)
    return image, y

def crop(image, y):    
    from_x = int(image.shape[1]*0.075)
    to_x = image.shape[1] - from_x
    
    from_y = int(image.shape[0]*0.075)
    to_y = image.shape[0] - from_y
    
    probability = random.randint(3,5)
    if probability==0:
        image = image[from_y:,from_x:,: ] 
    elif probability==1:
        image = image[:,from_x:,: ] 
    elif probability==2:
        image = image[from_y: ,: ] 
    elif probability==3:
        image = image[ :to_y, : ] 
    elif probability==4:
        image = image[ : , :to_x, ] 
    else :
        image = image[:to_y, :to_x, ]   
    return image, y

def data_augmentation_generator(path_features, labels_csv, batch_size, sample_size):
    img_rows, img_cols = 224, 224
    
    filenames = list( np.array([list(labels_csv.keys())[i] for i in range(2500) if (i+1)%25!=0 ]) )
    sample_size = 2400
    #filenames = list(labels_csv.keys())[:sample_size]
    
    while True:
        random.shuffle(filenames)
        for iterator in range(sample_size // batch_size):
            batch_features = np.zeros((batch_size, img_rows, img_cols, 3))
            batch_labels = np.zeros((batch_size, 50))
            
            for i in range(0, batch_size):
                # read picture
                filename = join(path_features, filenames[i + iterator * batch_size])
                pict = imread(filename, mode='RGB')
                
                #img = image.load_img(filename, target_size=(224, 224))
                #x = image.img_to_array(img)
                #x = np.expand_dims(x, axis=0)
                #pict = preprocess_input(x)
                
                coords = labels_csv[filenames[i + iterator * batch_size]]
                
                probability = random.randint(0,1)
                if probability == 1:
                    pict, coords = flip(pict, coords)
                
                probability = random.randint(0,1)
                if probability == 1:
                    pict, coords = rotate(pict, coords)
                
                probability = random.randint(0,1)
                if probability == 1:
                    pict, coords = crop(pict, coords)
                    
                pict = imresize(pict, (img_rows, img_cols), interp = 'bilinear')
                
                batch_features[i] = pict
                batch_labels[i][coords] = 1
            
            #batch_features = batch_features.astype('float32')# / 255.0
            batch_features = preprocess_input(batch_features)
            yield (batch_features, batch_labels)
            
from os.path import join
def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res

gt_dir = '/home/tamerlan/CV/task5/tests/00_gt'
train_img_dir = '/home/tamerlan/CV/task5/tests/00_input/train/images'
train_gt = read_csv(join(gt_dir, 'gt.csv'))

model = ResNet50(weights='imagenet')
#model = ResNet50(weights=None)
model.layers.pop()
x =  Dense(50, activation='softmax')(model.layers[-1].output)
            # x =  Dense(50, activation='softmax')(model.output)

final_model = Model(inputs = model.input, outputs = x)

sgd = SGD(lr=0.005)
final_model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])



count = len(train_gt)

final_model.fit_generator(data_augmentation_generator(train_img_dir, train_gt, 32, count), 
                                              steps_per_epoch=count//32,  
                                              epochs=10,
                                              verbose=1)
