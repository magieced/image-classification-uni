import os # TODO: What is this import for? It seems to be unused ~Erik
from os import read

from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
import numpy as np
from sklearn.utils import shuffle

from torchvision import transforms

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

def single_im_preprocessing(image:Image.Image,imsize=224): # changed size to 224 ~Erik
    """takes a single PIL image and scales it to imsize*imsize pixels(default: 224x224) and  blurs it with a gaussian kernel
    Args:
        image(PIL.Image.Image): the PIL Image to be preprocessed
        imsize(int): the sidelength to which the inputted image should be scaled
    Returns:
            the preprocessed Image, dtype=torch.tensor"""
    # Swapped from the previous implementation to transform images into C H W format ~Erik
    transform = transforms.Compose([
        transforms.Resize((imsize,imsize)),
        transforms.ToTensor()
    ])

    image = transform(image)

    gaus = GaussianBlur(5,1)
    if len(image.size()) == 2:
        image = image[:,:,None]
        image = image.repeat(1,1,3)
        image = image.convert("RGB")
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

    def __init__(self,train:bool,imsize:int):
        temppairs= im_labels_pair_getter()
        imspercent= len(temppairs)/100
        split= round(imspercent*80)
        data = list_im_preprocessing([(x[0]) for x in temppairs],imsize)
        # I'm sorry for this implementation
        # This is a temporary solution at best
        # Converts labels from str to int (python doesn't support multiclass lists) while also removing the negative class problem (crossEntropyLoss doesn't support negative classes)
        # ~Erik
        # TODO: Make the reclassing happen in a way that works with interface.py
        labels = [(int(x[1]) + 1) for x in temppairs]
        data = shuffle(data,random_state=0)
        labels = shuffle(labels,random_state=0)
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

def get_dataloaders(shuffled:bool=False, image_side_length:int=128):
    """creates and return one dataloader for training and one dataloader for validation
    Args:
        shuffled(bool): if true the dataloaders get shuffled, a.k.a. the order of the images with their labels gets randomized
        image_side_length(int): all images in the dataloaders will be resized to squares of this sidelength
    Returns:
        a training(first 80%) and a validation(last 20%) dataloader of the training images"""
    train_set = DataLoader(Imageset(train=True,imsize=image_side_length), batch_size=1, shuffle=shuffled)
    valid_set = DataLoader(Imageset(train=False,imsize=image_side_length), batch_size=1, shuffle=shuffled)
    return train_set,valid_set