"""
predict.py
Author: Kate Nguyen
Loading OCR model and predict labels for the images
"""
import keras
from keras.models import Model
import os
import sys
import numpy as np
from keras import applications
# import copy
import cv2 as cv
from random import shuffle
import json
from data_reader import Reader
from img_preprocessing import preprocess
from pixel_standardize import img_stdize, norm
from keras.models import load_model
from train import data_generator
import argparse

def main(args):
    # model_path = './logs/006-gpu/weights/model_final.h5'
    test_data_dir = "../dataset/test-data/original/"
    test_label_path = "../dataset/test-data/labels.txt"
    dataset_stats_path = './stdize_vals.txt'
    labels_dict_path = './labels_dict.json'
    
    if args.testdir:
        test_data_dir = args.testdir
    if args.testlabels:
        test_label_path = args.testlabels

    model_path = args.weight

    with open(labels_dict_path, 'r') as fp:
        labels_dict = json.load(fp)

    with open(test_label_path) as f:
        test_labels_list = f.readlines()

    np.random.shuffle(test_labels_list)
    if args.samples_num:
        samples_num = eval(args.samples_num)
        test_labels_list = test_labels_list[:samples_num]

    ### Generate images and labels
    image_array, ground_truth = test_image_generator(test_data_dir, test_labels_list, dataset_stats_path)
    
    ### Load model and predict
    model = load_model(model_path)
    # print(model.summary())
    predict = model.predict(image_array, verbose = 1)
    classes = list(np.argmax(predict, axis=1))
    samples_num = len(test_labels_list)
    true_count = 0
    
    ### Compare predictions against ground truth
    wrong_list = []
    key_list = list(labels_dict.keys()) 
    val_list = list(labels_dict.values())

    for i in range(samples_num):
        true_label = ground_truth[i]
        true_class = labels_dict[true_label]
        predict_class = classes[i]
        if predict_class == true_class:
            true_count += 1
        else:
            pred_char = key_list[val_list.index(predict_class)]
            wrong_tuple = (true_label, pred_char)
            wrong_list.append(wrong_tuple)
    
    ### Calculate acurracy and print out list of wrong predictions
    accuracy = true_count / samples_num
    print('Prediction accuracy: ', accuracy)
    print('List of wrong predictions (truth, prediction):')
    print(wrong_list)
    

def test_image_generator(test_data_dir, test_labels_list, dataset_stats_path):
    """
    Generate test images
    Outputs:
        image_array: a numpy array containing testing images 
        ground_truth: a list containing labels corresponding to the testing images
    """
    with open(dataset_stats_path, 'r') as f:
        stats = f.read().splitlines()

    mean_str = stats[0].split(':')[0]  ### stats[0] looks like '[val val val]:mean'
    mean_r, mean_g, mean_b = mean_str[1:-1].split()     ### leave out brackets and split
    rgb_means = [eval(mean_r), eval(mean_g), eval(mean_b)]

    std_str = stats[1].split(':')[0]
    std_r, std_g, std_b = std_str[1:-1].split()
    rgb_stds = [eval(std_r), eval(std_g), eval(std_b)]

    samples_num = len(test_labels_list)
    image_array = np.zeros((samples_num, 128,128,3))
    ground_truth = []

    for i in range(samples_num):
        image_name, label = test_labels_list[i].split()
        label = label[0]
        image = cv.imread(test_data_dir + image_name, cv.IMREAD_GRAYSCALE)
        image = preprocess(image)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        image = img_stdize(image, rgb_means, rgb_stds)

        image_array[i] = image
        ground_truth.append(label)

    return image_array, ground_truth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict labels of images using OCR model')
    parser.add_argument('--testdir', dest = 'testdir', help = 'Path to testing dataset')
    parser.add_argument('--testlabels', dest = 'testlabels', help = 'Path to testing labels file')
    parser.add_argument('--weight', required = True, dest = 'weight', help = 'Path to model file')
    parser.add_argument('--samples', dest = 'samples_num', help = 'Number of samples to predict. If not specified, use all samples in test directory.')
    args = parser.parse_args()
    main(args)