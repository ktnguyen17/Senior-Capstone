"""
train.py
Author: Kate Nguyen
Training code for OCR
"""
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import applications
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model
from keras.losses import SparseCategoricalCrossentropy as scce_loss
import os
import sys
import numpy as np
import copy
import cv2 as cv
from random import shuffle
from math import ceil
import json
from data_reader import Reader
from img_preprocessing import preprocess
from pixel_standardize import img_stdize, norm
import argparse

tf.get_logger().setLevel('ERROR')

def data_generator(data_dir, labels_list, dataset_stats_path, batch_size, labels_dict):
    """
    Data generator generates mini-batches to train model.
    Outputs:
        image_batch: a numpy array containing images in a mini-batch
        label_batch: a numpy array containing labels corresponding to the images in a mini-batch
    """

    ### Reading the pixel means and standard deviations of the dataset saved in a file
    with open(dataset_stats_path, 'r') as f:
        stats = f.read().splitlines()
     
    mean_str = stats[0].split(':')[0]                        ### stats[0] looks like '[val val val]:mean'
    mean_r, mean_g, mean_b = mean_str[1:-1].split()          ### leave out brackets and split
    rgb_means = [eval(mean_r), eval(mean_g), eval(mean_b)]

    std_str = stats[1].split(':')[0]
    std_r, std_g, std_b = std_str[1:-1].split()
    rgb_stds = [eval(std_r), eval(std_g), eval(std_b)]

    i = 0
    dataset_len = len(labels_list)
    shuffle(labels_list)
    image_batch = np.zeros((batch_size, 128,128,3))
    label_batch = np.zeros((batch_size,1))

    ### Generating mini-batches
    while True:
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(labels_list)

            image_name, label = labels_list[i].split()
            label = label[0]                                ### remove '\n' at the end

            image = cv.imread(data_dir + image_name, cv.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            image = img_stdize(image, rgb_means, rgb_stds)
            assert np.all(np.isfinite(image))
            
            image_batch[b] = image
            label_batch[b] = labels_dict[label]
            i = (i+1)%dataset_len

            assert np.all(np.isfinite(image_batch))
            assert np.all(np.isfinite(label_batch))

        yield image_batch, label_batch

def main(args):
    ### Default paths
    log_dir = "./logs/006-gpu/"
    data_dir = "../dataset/train-data/original/"
    label_path = "../dataset/train-data/labels.txt"
    test_data_dir = "../dataset/test-data/original/"
    test_label_path = "../dataset/test-data/labels.txt"
    weights_dir = log_dir + "weights/"
    dataset_stats_path = './stdize_vals.txt'
    input_weight = None
    gpu_num = 4
    labels_dict_path = './labels_dict.json'
    checkpoint_path = log_dir + 'checkpoint_weight.h5'
    cores_num = 8

    ### Change arguments if specified
    if args.logdir:
        log_dir = args.logdir
    if args.datadir:
        data_dir = args.datadir     
    if args.labels:
        label_path = args.labels
    if args.testdir:
        test_data_dir = args.testdir
    if args.testlabels:
        test_label_path = args.testlabels
    if args.weight:
        input_weight = args.weight
    if args.gpus:
        gpu_num = eval(args.gpus)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    if args.cores:
        cores_num = args.cores
    
    ### Hyperparameters
    num_class = 7356
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    val_split = 0.1
    
    ### Type of neural network
    if args.net == 'resnet':
        model = applications.ResNet50(include_top = True, weights = input_weight, input_shape = (128, 128, 3), classes=num_class)
    elif args.net == 'inception':
        model = applications.InceptionV3(include_top = True, weights = input_weight, input_shape = (128, 128, 3), classes=num_class)

    ### Train with GPU or CPU
    if args.procunit == 'GPU':
        threading = False
        if gpu_num >= 2:
            model = multi_gpu_model(model, gpus=gpu_num)
    elif args.procunit == 'CPU':
        tf.config.threading.set_intra_op_parallelism_threads(cores_num)
        tf.config.threading.set_inter_op_parallelism_threads(cores_num)
        threading = True

    ### Logging
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, verbose=1)
    
    ### Optimizer and loss function
    optimizer = Adam(learning_rate=learning_rate, clipnorm = 1)
    model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy, metrics=['sparse_categorical_accuracy'])
    
    ### Read label file
    with open(label_path) as f:
        labels_list = f.readlines()

    ### Read classes dictionary
    with open(labels_dict_path, 'r') as fp:
        labels_dict = json.load(fp)

    ### Train with a subset of dataset
    labels_list = labels_list[:750000]

    np.random.seed(930606)              ### Use random seed for reproducibility
    np.random.shuffle(labels_list)

    ### Train-validation split
    num_train = int(len(labels_list) * (1-val_split))
    num_val = len(labels_list) - num_train
    labels_train = labels_list[:num_train]
    labels_val = labels_list[num_train:]
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    ### Generate train and val datasets
    train_generator = data_generator(data_dir, labels_train, dataset_stats_path, batch_size, labels_dict)
    val_generator = data_generator(data_dir, labels_val, dataset_stats_path, batch_size, labels_dict)

    ### Resume training from a checkpoint
    if args.resume == 'True':
        model.load_weights(checkpoint_path)
    
    ### Training
    model.fit_generator(generator=train_generator, steps_per_epoch=ceil(num_train/batch_size), epochs = epochs, validation_data=val_generator, validation_steps=ceil(num_val/batch_size), initial_epoch=0, callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    ### Save final weight and full model
    model.save_weights(weights_dir + 'trained_weights_final.h5')
    model.save(weights_dir + 'model_final.h5')

    ### Testing and evaluation
    Print('Testing')
    
    with open(test_label_path) as f:
        test_labels_list = f.readlines()

    np.random.shuffle(test_labels_list)
    test_labels_list = test_labels_list[:100000]
    test_generator = data_generator(test_data_dir, test_labels_list, dataset_stats_path, batch_size, labels_dict)

    scores = model.evaluate_generator(generator=test_generator, steps= 512,  callbacks=[logging, checkpoint, reduce_lr, early_stopping], use_multiprocessing=threading, verbose=1, workers = 1, max_queue_size = 15)
    print('Test loss: ', scores[0])
    print('Test accuracy: ', scores[1])

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the OCR.')
    parser.add_argument('--net', dest = 'net',required= True, help = 'Choose type of neural network: resnet or inception')
    parser.add_argument('--procunit', dest = 'procunit',required = True, help = 'Choose between GPU and CPU')
    parser.add_argument('--logdir', dest = 'logdir', help = 'Path to log directory')
    parser.add_argument('--labels', dest = 'labels', help = 'Path to training labels file')
    parser.add_argument('--datadir', dest = 'datadir', help = 'Path to training dataset')
    parser.add_argument('--testdir', dest = 'testdir', help = 'Path to testing dataset')
    parser.add_argument('--testlabels', dest = 'testlabels', help = 'Path to testing labels file')
    parser.add_argument('--gpus', dest = 'gpus', help = 'Number of GPUs, only when --procunit set to GPU')
    parser.add_argument('--weight', dest = 'weight', help = 'Path to pretrained weight file')
    parser.add_argument('--resume', dest = 'resume', required = True, help ='True: resume training from checkpoint. False: train from scracth.')
    parser.add_argument('--cores', dest = 'cores', help = 'Number of cores')
    parser.add_argument('--checkpoint', dest = 'checkpoint', help = 'Path to model checkpoint file, only specify when --resume is True')
    args = parser.parse_args()

    if args.net not in ['resnet', 'inception']:
        print('--net must be resnet or inception\nAborted')
        sys.exit()
    if args.procunit not in ['CPU', 'GPU']:
        print('--procunit must be CPU or GPU\nAborted')
        sys.exit()
    if args.resume not in ['True', 'False']:
        print('--resume must be True or False\nAborted')
        sys.exit()
    if args.procunit == 'CPU':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main(args)
