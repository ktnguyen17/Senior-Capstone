# An Integrated Model for Offline Handwritten Chinese Character Recognition Based on Convolutional Neural Networks
### Author: Dung Nguyen (Kate)  
  
[Paper](https://drive.google.com/file/d/1z-ackF1RNYTDUD3lwj8wXjcC94-2gIYg/view?usp=sharing)  
[Software Demo](https://youtu.be/SYP4UYh4N88)

## Software Architecture Diagram
<img src="https://code.cs.earlham.edu/ktnguyen17/senior-capstone/-/raw/master/SAD.png" width="512">

## Dataset
CASIA Online and Offline Chinese Handwriting Databases (http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)  
Dataset directory in Layout: /mounts/layout/ktnguyen17/dataset/

## Dependencies
* Tensorflow (1.15.0)
* Keras (2.3.1)
* OpenCV (4.2.0)
* Pillow (7.1.1)

## Usage
### Read data from the dataset
Read the GNT files and write the images and corresponding labels to disk.  
`python data_reader.py --datadir [path-to-GNT-dataset]` 

### Label encoder
Map the 7356 character classes to integers in range [0, 7355] and save the dictionary to a json file.  
`python label_encoder.py`

### Pixel Standardization
Calculate the pixel mean and standard deviation of all the images in the dataset.  
`python pixel_standardize.py`

### Training
Train the CNN models.  
`python train.py`
* `--net [resnet or inception]` **(required)** Choose the type of neural network.
* `--procunit [GPU or CPU]` **(required)** Choose the type processing unit to train with.
* `--resume [True or False]` **(required)** Choose True to train from scratch, choose False to resume training from the previous checkpoint.
* `--checkpoint [path]` Specify path to the checkpoint file, only if --resume is set to True.
* `--logdir [path]` Specify log directory.
* `--datadir [path]` Specify training data directory.
* `--labels [[path]` Specify path to training labels file.
* `--testdir [path]` Specify testing data directory.
* `--testlabels [path]` Specify path to testing labels file.
* `--gpus [integer]` Number of GPUs, only if --procunit is set to GPU.
* `--cores [integer]` Number of cores for threading.
* `--weight [path]` Path to pretrained weight file.

### Classify and detect errors
After trainning, load the model and classify the characters. Print out a list containing (true_label, wrong_prediction) tuples.  
`python prediction.py`
* `--weight [path]` **(required)** Path to model file (*.h5).
* `--samples [integer]` Specify number of samples to classify. If not specified, use all samples in the test directory.
* `--testdir [path]` Specify testing data directory.
* `--testlabels [path]` Specify path to testing labels file.
