"""
data_reader.py
Author: Kate Nguyen
Read data from GNT files and save images to disk.
Reference: The load_gnt_file function was adapted from this source https://www.programmersought.com/article/72761667582/
"""
import struct
from codecs import decode
from PIL import ImageOps, Image, ImageShow
import numpy as np
import PIL.ImageOps
import os
import sys
import argparse

class Reader:
    def load_gnt_file(self, filename, img_out_dir, label_path, count):
        
        with open(filename, "rb") as f:
            label_file = open(label_path, 'a')
            while True:
                packed_length = f.read(4)
                if packed_length == b'':
                    break
                length = struct.unpack("<I", packed_length)[0]
                raw_label = struct.unpack(">cc", f.read(2))
                width = struct.unpack("<H", f.read(2))[0]
                height = struct.unpack("<H", f.read(2))[0]
                photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
                label = decode(raw_label[0] + raw_label[1], encoding="GB18030")
                image = Image.fromarray((np.array(photo_bytes).reshape(height, width)).astype('uint8'))
                img_name = str(count) + ".png"
                img_out_path = img_out_dir + img_name
                image.save(img_out_path)
                label_line = img_name + ' ' + label + '\n'
                label_file.write(label_line)
                count += 1
                
            return count


def main(args):
    reader = Reader()
    datadir = '/mounts/layout/ktnguyen17/dataset/'
    if args.datadir:
        datadir = args.datadir
    dirs = [datadir + 'train-data/gnt/', datadir + 'test-data/gnt/']
    count = 0
    for dir_path in dirs:
        files = os.listdir(dir_path)
        length = len(files)
        if 'train-data' in dir_path:
            label_path = datadir + 'train-data/labels.txt'
            img_out_dir = datadir + 'train-data/original/'
        elif 'test-data' in dir_path:
            label_path = datadir + 'test-data/labels.txt'
            img_out_dir = datadir + 'test-data/original/'
        for index, file in enumerate(files):
            file_path = dir_path + file
            count = reader.load_gnt_file(file_path, img_out_dir, label_path, count)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data from GNT files')
    parser.add_argument('--datadir', dest = 'datadir', help = 'Path to dataset (contains both train and test data)')
    args = parser.parse_args()
    main(args)
