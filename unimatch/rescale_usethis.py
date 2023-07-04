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

path = '/home/jsh/neurips/ours_data_occlusion/dolphins-show_512/'
files = sorted(glob.glob((path + '/*.jpg')))

save_path = '/home/jsh/neurips/Video-P2P-nana/data/images/dolphins-show_512/'

for idx, f in enumerate(files):
    image = Image.open(f)
    image_resize = image.resize((512, 512))
    image_resize.save(save_path + str(idx).zfill(4) + ".png", "png")
    


    



