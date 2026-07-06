import os
from os import read

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
import numpy as np

def im_labels_pair_getter(folder="21ClassDataset/",label_file="labels_21ClassDataset.csv"):
    labels=open(folder+label_file)
    labels.__next__()
    pairs=[]
    while True:
        line = labels.readline()
        if line == '':
            break
        parts = line.split(',')
        im = Image.open(folder+parts[0])
        pairs += [[im,parts[1]]]
    return pairs

def single_im_preprocessing(image:Image.Image,imsize=128):
    """takes a single PIL image and scales it to imsize*imsize pixels(default: 128x128) and  blurs it with a gaussian kernel
    Args:
        image(PIL.Image.Image): the PIL Image to be preprocessed
        imsize(int): the sidelength to which the inputted image should be scaled
    Returns:
            the preprocessed Image, dtype=torch.tensor"""
    image = torch.tensor(np.array(image.resize((imsize,imsize))))
    gaus = GaussianBlur(5,1)
    image = gaus.forward(image)
    return image

def list_im_preprocessing(images:list[Image.Image],imsize=128):
    """applies single_im_preprocessing over teh given list of images, scaling them ro imsize*imsize
    Args:
        images(list[PIL.Image.Image]): the PIL Images to be preprocessed
        imsize(int): the sidelength to which the inputted images should be scaled
    Returns:
            the preprocessed Images, dtype=list[torch.tensor]"""
    for i in range(len(images)):
        images[i]=single_im_preprocessing(images[i],imsize)
    return images

class Imageset(torch.utils.data.Dataset):

    def __init__(self,train:bool):
        temppairs= im_labels_pair_getter()
        imspercent= len(temppairs)/100
        split= round(imspercent*80)
        data = list_im_preprocessing([(x[0]) for x in temppairs])
        labels = [x[1] for x in temppairs]
        if train:
            self.data = data[:split]
            self.labels = labels[:split]
        else:
            self.data = data[split:]
            self.labels = labels[split:]
            #self.cat_dog_label =

    def __getitem__(self, item):
        return self.data[item],torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.data)
c = Imageset(True)

def getdataloaders():
    """creates and return one dataloader for training and one dataloader for validation"""
    train_set = DataLoader(Imageset(train=True),batch_size=1)
    valid_set = DataLoader(Imageset(train=False),batch_size=1)
    return train_set,valid_set

