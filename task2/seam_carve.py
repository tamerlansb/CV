from math import sqrt 
import numpy as np

def min2_p(a,b):
    if a<=b :
        return 0
    else :
        return 1
    
def min2_n(a,b):
    if a<=b :
        return -1
    else :
        return 0

def min3(a,b,c):
    if a<=b and a<=c:
        return -1
    elif b<=c :
        return 0
    else :
        return 1

def seam_carve(img, mode_param, mask):
    direction,mode = mode_param.split()
    Y = np.dot(img, np.array([0.299, 0.587, 0.114]))
    
    grad_x = np.roll(Y,1,0) - np.roll(Y,-1,0)
    grad_x[0] = Y[1,:] - Y[0,:]
    grad_x[grad_x.shape[0]-1,:] = Y[grad_x.shape[0]-1,:] - Y[grad_x.shape[0]-2,:]
    grad_y = np.roll(Y,-1,1) - np.roll(Y,1,1)
    grad_y[:,0] = Y[:,1] - Y[:,0]
    grad_y[:,grad_y.shape[1]-1] = Y[:,grad_x.shape[1]-1] - Y[:,grad_x.shape[1]-2]
    grad = (grad_x**2 + grad_y**2)**(0.5)

    if mask is not None:
        grad = grad + mask*1000.0
    if direction != 'horizontal':
        grad = grad.transpose()
        img = img.transpose((1,0,2))
        if mask is not None:
            mask = mask.transpose()
            
        
    mask_seam = np.zeros((grad.shape[0],grad.shape[1]))
    min_seam = np.zeros( (grad.shape[0],grad.shape[1]))
    if mode=='shrink':
        w = grad.shape[1] - 1
    else :
        w = grad.shape[1] + 1
    result_img = np.zeros((grad.shape[0],w,3))
    result_mask = np.zeros((grad.shape[0],w))
    
    min_seam[0,...] = grad[0,...]
    for i in range(1, grad.shape[0]):
        for j in range(grad.shape[1]):
            if j == 0:
                min_seam[i,j] = grad[i,j] + min(min_seam[i-1,j],min_seam[i-1,j+1])
            elif j == grad.shape[1]-1: 
                min_seam[i,j] = grad[i,j] + min(min_seam[i-1,j-1],min_seam[i-1,j] )
            else:
                min_seam[i,j] = grad[i,j] + min(min_seam[i-1,j-1],min_seam[i-1,j],min_seam[i-1,j+1])
          
    j_min = 0
    for j in range(grad.shape[1]):
        if min_seam[grad.shape[0]-1,j] < min_seam[grad.shape[0]-1,j_min]:
            j_min = j
            
    for i in range(grad.shape[0]-1,-1,-1):
        mask_seam[i,j_min] = 255 
        
        ############ Сжатие или расширение изображения и маски(если есть) ################
        result_img[i,:j_min] = img[i,:j_min]
        if mask is not None :
            result_mask[i,:j_min] = mask[i,:j_min]
        if mode=='shrink':
            result_img[i,j_min:w] = img[i,j_min+1:w+1]
            if mask is not None :
                result_mask[i,j_min:w] = mask[i,j_min+1:w+1]
        elif j_min !=  grad.shape[1] -1 :
            result_img[i,j_min+2:w] = img[i,j_min+1:grad.shape[1]]
            result_img[i,j_min+1] = (img[i,j_min] + img[i,j_min+1])/2
            if mask is not None :
                result_mask[i,j_min+2:w] = mask[i,j_min+1:grad.shape[1]]
                result_mask[i,j_min+1] = (mask[i,j_min] + mask[i,j_min+1])/2
        ##################################################################################
        if i != 0: 
            if j_min == grad.shape[1] -1:
                j_min = j_min + min2_n(min_seam[i-1, j_min -1 ],min_seam[i-1, j_min])
            elif j_min == 0:
                j_min = j_min + min2_p(min_seam[i-1, j_min],min_seam[i-1, j_min +1]) 
            else :
                j_min = j_min + min3(min_seam[i-1, j_min-1],min_seam[i-1, j_min],min_seam[i-1, j_min +1] )
    if direction != 'horizontal':
        mask_seam = mask_seam.transpose()
        result_img = result_img.transpose(1,0,2)
        result_mask = result_mask.transpose()
    return result_img,result_mask, mask_seam

