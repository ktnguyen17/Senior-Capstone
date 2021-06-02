"""
img_preprocessing.py
Author: Kate Nguyen
Apply image pre-processing functions on the images.
"""
import cv2 as cv
import numpy as np
from PIL import ImageOps, Image, ImageShow
from matplotlib import pyplot as plt
import os
from math import ceil

def smooth(image): #input numpy array
    # Gaussian filtering
    blur = cv.GaussianBlur(image, (3,3), 0) 
    return blur

def binarize(image):    #take numpy array as input
    # Otsu's thresholding
    thres_val, otsu_img = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return otsu_img

def size_normalize(image): #take numpy array as input
    image = Image.fromarray(image)
    image = image.resize((128, 128))
    return np.array(image)

def resize_and_padding(image):
    """
    Normalize the shape of the images to (128,128) and avoid stretching the images.
    First, enlarge the image so that the larger dimension becomes 128.
    Next, increase the other dimension to 128 by padding, which is adding pixels with background color to both sides of the dimension.
    """
    ### cv.imread returns shape (height, width)
    enlarge_ratio = 128/max(image.shape)
    new_dim = (ceil(image.shape[0]* enlarge_ratio), ceil(image.shape[1]*enlarge_ratio))
    image = Image.fromarray(image).resize((new_dim[1],new_dim[0])) ### resize function takes (width, height) as arg
    image = np.array(image)
    top_pad = int((128-image.shape[0])/2)
    bot_pad = int((128-image.shape[0]) - top_pad)
    left_pad = int((128-image.shape[1])/2)
    right_pad = int((128-image.shape[1]) - left_pad)
    image= cv.copyMakeBorder(image,top_pad,bot_pad,left_pad,right_pad,cv.BORDER_CONSTANT,value=[255,255,255])
    return image

def denoise(image):
    denoised = cv.fastNlMeansDenoising(image, None, 3, 7, 21)
    return denoised

def thinning(image):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv.erode(image, kernel, iterations = 1)
    return erosion

# def preprocess(image):
#     image = smooth(image)
#     image = denoise(image)
#     image = resize_and_padding(image)
#     image = binarize(image)
#     image = thinning(image)
#     return image

def preprocess(image):
    image = cv.GaussianBlur(image, (3,3), 0)  ### smooth
    image = cv.fastNlMeansDenoising(image, None, 3, 7, 21)  ### denoise
    image = resize_and_padding(image)
    thres_val, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)    ### binarize
    kernel = np.ones((3,3), np.uint8)
    image = cv.erode(image, kernel, iterations = 1) ### thinning
    return image

### Only for testing
def main():
    dirs = ['/mounts/layout/ktnguyen17/dataset/train-data/original/']

    for dir_path in dirs:
        files = os.listdir(dir_path)
        if 'train-data' in dir_path:
            img_out_dir = '/mounts/layout/ktnguyen17/dataset/train-data/preprocessed/'
        elif 'test-data' in dir_path:
            img_out_dir = '/mounts/layout/ktnguyen17/dataset/test-data/preprocessed/'
        
        for index, file in enumerate(files):
            file_path = dir_path + file
            image = cv.imread(file_path, cv.IMREAD_UNCHANGED)
            image = preprocess(image)
            image = Image.fromarray(image)
            img_out_path = img_out_dir + file
            image.save(img_out_path)
            
if __name__ == '__main__':
    main()
