"""
pixel_standardize.py
Author: Kate Nguyen
Calculate the pixel means and standard deviations and standardize pixel values of the dataset.
The standardized dataset has a zero mean and unit variance.
"""
from PIL import Image
import numpy as np
from img_preprocessing import preprocess
import cv2 as cv
import os
import sys


def cal_dataset_stat(data_dir, label_path): ### Find means and stds of each rgb channel of dataset
    """
    Calculate means and standard deviations of all 3 channels
    Output:
        rgb_mean: an array containing means of 3 channels
        rgb_std: an array containing standard deviations of 3 channels
    """
    with open(label_path) as f:
        labels_list = f.readlines()
    pixel_num = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    for i in range(len(labels_list)):
        image_name, label = labels_list[i].split()
        file_path = data_dir + image_name
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        image = preprocess(image)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        image = np.asarray(image)
        image = image.astype('float32')
        #im = im/255.0
        pixel_num += (image.size/3)         ### channel number = 3
        channel_sum += np.sum(image, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(image), axis=(0, 1))
    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

    return rgb_mean, rgb_std

def img_stdize(image, means, stds):         ### image input is a np.array
    """
    Standardize the pixel values
    Output: a standardized image
    """
    means = np.array(means)
    stds = np.array(stds)
    assert np.all(np.isfinite(stds))
    assert 0 not in stds
    pixels = image.astype('float32')
    pixels = (pixels - means) / stds
    assert np.all(np.isfinite(pixels))
    return pixels

def norm(image):
    pixels = image.astype('float32')
    pixels /= 255.0
    return pixels


### test
def main():
    data_dir = '/mounts/layout/ktnguyen17/dataset/train-data/original/'
    label_path = '/mounts/layout/ktnguyen17/dataset/train-data/labels.txt'
    rgb_mean, rgb_std = cal_dataset_stat(data_dir, label_path)
    out_file = open('stdize_vals.txt', 'w')
    mean = str(rgb_mean) + ':mean\n'
    std = str(rgb_std) + ':standard deviation'
    out_file.write(mean)
    out_file.write(std)
    out_file.close()

if __name__ == '__main__':
    main()