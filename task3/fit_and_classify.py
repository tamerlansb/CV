import numpy as np
from sklearn.svm import SVC
from scipy.signal import convolve2d
from scipy.misc import imresize
from numpy import linalg as LA
from math import sqrt


import numpy as np
from sklearn.svm import SVC
from scipy.signal import convolve2d
from scipy.misc import imresize
from numpy import linalg as LA
from math import sqrt


def extract_hog(img):
    def max3(a,b,c):
        if a>=b and a>=c:
            return 0
        elif b>=c :
            return 1
        else :
            return 2
        
        
    def get_hog_vector(orientation_block, magnitude_block):
        count_part_pi = 9
        hog_vector = np.zeros(count_part_pi)
        for i in range(orientation_block.shape[0]):
            for j in range(orientation_block.shape[1]):
                max_slice_num = max3(magnitude_block[i, j, 0], magnitude_block[i ,j, 1], magnitude_block[i, j, 2])
                freq_num = round(9*orientation_block[i,j][max_slice_num] / 180)
                hog_vector[int(freq_num)%9] += magnitude_block[i,j][max_slice_num]
        return hog_vector
    
    
    img = imresize(img, (48,48))
    
    #kernels
    #Dx = np.array([[-1, 0, 1]])
    #Dy = np.array([[-1], [0], [1]])
    Dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Dy = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    
    #compute gradient along X-axes
    gx = np.zeros((img.shape[0], img.shape[1], 3))
    gx[...,0] = convolve2d(img[...,0] ,Dx, 'same')
    gx[...,1] = convolve2d(img[...,1] ,Dx, 'same')
    gx[...,2] = convolve2d(img[...,2] ,Dx, 'same')
    
    #compute gradient along Y-axes
    gy = np.zeros((img.shape[0], img.shape[1], 3))
    gy[...,0] = convolve2d(img[...,0], Dy, 'same')
    gy[...,1] = convolve2d(img[...,1], Dy, 'same')
    gy[...,2] = convolve2d(img[...,2], Dy, 'same')
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (abs(np.arctan2(gy, gx)) * 180 / np.pi) 
    
    block = np.zeros((6,6,9))
    full_hog_vector = np.empty(0)
    epsilon = 0.00001
    
    for i in range(6):
        for j in range(6):
            orientation_block = orientation[i*8: (i+1)*8, j*8: (j+1)*8,:]
            magnitude_block   =   magnitude[i*8: (i+1)*8, j*8: (j+1)*8,:]
            block[i,j] = get_hog_vector(orientation_block, magnitude_block)
    
    for i in range(5):
        for j in range(5):
            con_v = np.concatenate((block[i, j], block[i + 1, j], block[i, j + 1], block[i + 1, j + 1]))
            con_v = con_v / sqrt(LA.norm(con_v)**2 + epsilon)
            full_hog_vector = np.concatenate((full_hog_vector, con_v))
    return full_hog_vector


def fit_and_classify(train_features, train_labels, test_features):
    model = SVC(kernel='rbf', C=1580,gamma=0.451)
    model.fit(train_features, train_labels)
    y = model.predict(test_features)
    return y