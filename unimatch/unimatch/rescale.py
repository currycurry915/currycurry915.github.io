#rescale
import os 
from PIL import Image
import glob
import ntpath
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.utils.data
import random
from os.path import join, splitext, basename
import math


def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale <= 1 and width_scale <=1:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img


path = '/mnt/storage1/jhkim/landmark/gldv2_micro/test/'

files = glob.glob((path + '/*.jpg'))

save_path = '/mnt/storage1/image_bridge/landmark_dataset/test/'

a = 0
for f in files:
    for idx, file in enumerate(files):
        fname, ext = os.path.splitext(file)
        bn = ntpath.basename(fname)
        
        if ext in ['.jpg', '.png', '.gif']:
            im = Image.open(file)

            croped_image = center_crop(im, 256, 768)

        croped_image.save(save_path + str(a).zfill(6) + '.png')

        a += 1

    break


    



