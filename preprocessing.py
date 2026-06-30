from os import read

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import GaussianBlur

def im_labels_pair_getter(folder="images/"):
    labels=open(folder+"labels.csv")
    labels.__next__()
    pairs=[]
    while True:
        line = labels.readline()
        if line == '':
            break
        parts = line.split(',')
        im = Image.open(folder+parts[0])
        pairs += [(im,parts[1])]
    return pairs

def single_im_preprocessing(image:Image.Image):
    image = image.resize((256,256))
    gaus = GaussianBlur(5,1)
    image = gaus.forward(image)
    return image

