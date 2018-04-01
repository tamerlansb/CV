#aligned_img, (b_row, b_col), (r_row, r_col) = align(img, g_coord)
import numpy as np
from math import sqrt
from skimage import measure

def EnvironsCheck(Im1,Im2,y,x,compress_coef):
    mse_min = float("+inf")
    x_shift = 0
    y_shift = 0
    T = Im1
    for j in range(y - compress_coef,y + compress_coef):
        for i in range(x - compress_coef,x + compress_coef):
            S = np.roll(np.roll(Im2,i,1),j,0)
            mse = measure.compare_mse(T,S)
            if (mse < mse_min):
                mse_min = mse
                x_shift = i
                y_shift = j
    return y_shift,x_shift

def compress(img):
    return measure.block_reduce(img,(2,2))

def min_coord_mse(Im1,Im2):
    mse_min = float("+inf")
    x_shift = 0
    y_shift = 0
    for x in range(-15,15):
        for y in range(-15,15):
            T = Im1
            S = np.roll(np.roll(Im2,x,1),y,0)
            mse = measure.compare_mse(T,S)
            if (mse < mse_min):
                mse_min = mse
                x_shift = x
                y_shift = y
    return y_shift,x_shift

def align(im, g_coord):
    im = im * 255
    h = im.shape[0] // 3
    deltaX = h*15 // 100
    deltaY = im.shape[1]*15//100
    B = im[      deltaX:  h - deltaX,  deltaY :im.shape[1] -deltaY]
    G = im[  h + deltaX:2*h - deltaX,  deltaY :im.shape[1] -deltaY]
    R = im[2*h + deltaX:3*h - deltaX,  deltaY :im.shape[1] -deltaY]
    
    if (G.shape[0]>512 or G.shape[1]>512):
        G_small = G
        R_small = R
        B_small = B
        compress_coef = 1
        while (R_small.shape[0]>512 and R_small.shape[1]>512):
            compress_coef = compress_coef * 2
            R_small = compress(R_small)
            G_small = compress(G_small)
            B_small = compress(B_small)
        r = min_coord_mse(G_small,R_small)
        r_0 = r[0]*compress_coef
        r_1 = r[1]*compress_coef
        r = EnvironsCheck(G,R,r_0,r_1,compress_coef//2)
        
        b = min_coord_mse(G_small,B_small)
        b_0 = b[0]*compress_coef
        b_1 = b[1]*compress_coef
        b = EnvironsCheck(G,B,b_0,b_1,compress_coef//2)
    else :
        r = min_coord_mse(G,R)
        b = min_coord_mse(G,B)
    
    B_aligned = np.roll(B,b[0],0)
    B_aligned = np.roll(B_aligned,b[1],1)

    R_aligned = np.roll(R,r[1],1)
    R_aligned = np.roll(R_aligned,r[0],0)
    
    aligned_img = np.zeros( (R.shape[0],R.shape[1],3), dtype=np.uint8)
    aligned_img[...,0] = R_aligned
    aligned_img[...,1] = G
    aligned_img[...,2] = B_aligned
    return aligned_img,  (g_coord[0] - b[0] - h, g_coord[1] - b[1] ), (g_coord[0] - r[0] + h, g_coord[1] - r[1] )
