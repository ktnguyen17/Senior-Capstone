"""
label_encoder.py
Author: Kate Nguyen
Encode classes. Map each class (a class is a character) to an integer and store the dictionary in a json file.
"""
import json

def encoder(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    labels_dict = {}
    dict_index = 0
    for line in lines:
        label = line.split()[1]
        label = label[0]
        if label not in labels_dict:
            labels_dict[label] = dict_index
            dict_index += 1
    return labels_dict

if __name__ == '__main__':
    label_path = '/mounts/layout/ktnguyen17/dataset/train-data/labels.txt'
    labels_dict = encoder(label_path)
    print('number of classes ', len(labels_dict))

    with open('labels_dict.json', 'w') as fp:
        json.dump(labels_dict, fp)
