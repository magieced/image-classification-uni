import random
from os import error

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

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

def single_im_preprocessing(image:Image.Image,imsize=224)->torch.Tensor: # changed size to 224 ~Erik
    """takes a single PIL image and scales it to imsize*imsize pixels(default: 224x224) and  blurs it with a gaussian kernel
    Args:
        image(PIL.Image.Image): the PIL Image to be preprocessed
        imsize(int): the sidelength to which the inputted image should be scaled
    Returns:
            the preprocessed Image, dtype=torch.Tensor"""
    image = image.convert("RGB")
    gaus = GaussianBlur(5, 1)
    image = gaus.forward(image)
    image = image.resize((imsize,imsize))
    imarray = np.array(image)
    imtensor = torch.tensor(imarray)
    imtensor = imtensor.permute(2,0,1) #[H,W,C]->[C,H,W]
    return imtensor/255

def list_im_preprocessing(images:list[Image.Image],imsize=128)->list[torch.Tensor]:
    """applies single_im_preprocessing over teh given list of images, scaling them ro imsize*imsize
    Args:
        images(list[PIL.Image.Image]): the PIL Images to be preprocessed
        imsize(int): the sidelength to which the inputted images should be scaled
    Returns:
            the preprocessed Images, dtype=list[torch.Tensor]"""
    result:list[torch.Tensor] = [None] * len(images) #type: ignore[list-item]
    for i in tqdm(range(len(images)),desc="preprocessing for the dataset"):
        result[i]=single_im_preprocessing(images[i],imsize)
    return result

def image_hide_and_seek(image:torch.Tensor,patches_side:int,patches_length:int)->torch.Tensor:
    for i in range(patches_side):
        for j in range(patches_side):
            if random.choice([0,1]) == 0: #0 for not visible 1 for visible
                hdim:int=i*patches_length
                wdim:int=j*patches_length
                image[:,hdim:hdim+patches_length,wdim:wdim+patches_length] = 0
    return image
class PreprocessedPairStorage():
    def __init__(self,imsize:int,):
        temppairs = im_labels_pair_getter()
        imspercent = len(temppairs) / 100
        data = list_im_preprocessing([(x[0]) for x in temppairs], imsize)
        labels = [int(x[1].replace('-1', '20')) for x in temppairs]

        self.data:list[torch.Tensor] = shuffle(data, random_state=0)
        self.labels:list[int] = shuffle(labels, random_state=0)
        self.split:int = round(imspercent * 80)
        self.augmented:bool=False

    def augment(self,factor:int,val_destructive:bool=True,patches:int=16):
        """augments the data through the hide-and-seek algorithm
        (dividing the image into 16 patches and blacking out each one with a 50% chance)
        this is done factor times per image to expand the data factor times.
        It is strongly recommended not to build a validation set with augmented images
        Args:
            factor(int):by what amount the dataset should be extended, if under 1 this function doesn't do anything
            val_destructive(bool): if true destroys the validation part of the dataset,
             then augments the training part(this leads the inability to create a new validation set with this data,
             but should reduce runtime and ram-usage of this data by ~20%)
            patches(int): how many patches to divide the images into for the hide-and-seek augmentation, must have an int square root,
             otherwise the function automatically round down to the nearest int that fulfills these conditions(default: ``16``)
        """
        if factor<1 or self.augmented:
            return
        else:
            patch_side:int= int(patches**0.5)
            patch_side_length:int = int(self.data[1].size()[1]/patch_side)
            if not patch_side_length*patch_side == self.data[1].size()[1]:
                raise error("you are trying to use a number of patches that can't be distributed equally over the size you are scaling the image to, please change the images side-length the number of patches or set the augment-factor to 0")
            if val_destructive:
                new_data: list[torch.Tensor] = [None] * self.split * factor  # type:ignore[list-item]
                new_labels: list[int] = [None] * self.split * factor  # type:ignore[list-item]
                for index in tqdm(range(len(new_data)), desc="creating augmented data"):
                    old_index = index // factor
                    new_data[index] = image_hide_and_seek(self.data[old_index],patch_side,patch_side_length)
                    new_labels[index] = self.labels[old_index]
                self.split = len(new_data)
            else:
                new_data:list[torch.Tensor] = [None]*len(self.data)*factor #type:ignore[list-item]
                new_labels:list[int] = [None]*len(self.labels)*factor #type:ignore[list-item]
                for index in tqdm(range(len(new_data)),desc="creating augmented data"):
                    old_index = index//factor
                    new_data[index] = image_hide_and_seek(self.data[old_index],patch_side,patch_side_length)
                    new_labels[index] = self.labels[old_index]
                self.split=self.split*factor
            self.data = new_data
            self.labels = new_labels
            self.augmented = True



class Imageset(torch.utils.data.Dataset):

    def __init__(self,train:bool,storage:PreprocessedPairStorage):

        if train:
            self.data = storage.data[:storage.split]
            self.labels = storage.labels[:storage.split]
        else:
            self.data = storage.data[storage.split:]
            self.labels = storage.labels[storage.split:]
            #self.cat_dog_label =

    def __getitem__(self, item):
        return self.data[item],torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.data)

class ImagesetFull(torch.utils.data.Dataset):

    def __init__(self,storage:PreprocessedPairStorage):

            self.data = storage.data
            self.labels = storage.labels

    def __getitem__(self, item):
        return self.data[item],torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.data)

def get_dataloaders(shuffled:bool=False, image_side_length:int=224, augment_factor:int=0):
    """creates and return one dataloader for training and one dataloader for validation
    Args:
        shuffled(bool): if true the dataloaders get shuffled, a.k.a. the order of the images with their labels gets randomized (default:``False``)
        image_side_length(int): all images in the dataloaders will be resized to squares of this sidelength (default:``224``)
        augment_factor(int):if 1 or greater multiplies the amount of training data by this number using hide-and-seek data augmentation, if 0 does nothing
         (default:``0``)
    Returns:
        a training(first 80%[possibly increased trough augment_factor]) and a validation(last 20%) dataloader of the training images"""
    data_storage = PreprocessedPairStorage(image_side_length)

    valid_set = DataLoader(Imageset(train=False,storage=data_storage), batch_size=1, shuffle=shuffled)
    data_storage.augment(augment_factor)
    train_set = DataLoader(Imageset(train=True,storage=data_storage), batch_size=32, shuffle=shuffled)
    return train_set,valid_set
def get_one_dataloader(shuffled:bool=False, image_side_length:int=224, augment_factor:int=0):
    """creates and return one dataloader for training and one dataloader for validation
        Args:
            shuffled(bool): if true the dataloader gets shuffled, a.k.a. the order of the images with their labels gets randomized (default:``False``)
            image_side_length(int): all images in the dataloader will be resized to squares of this sidelength (default:``224``)
            augment_factor(int):if 1 or greater multiplies the amount of data by this number using hide-and-seek data augmentation, if 0 does nothing
             (default:``0``)
        Returns:
            a training(first 80%[possibly increased trough augment_factor]) and a validation(last 20%) dataloader of the training images"""
    data_storage = PreprocessedPairStorage(image_side_length)
    data_storage.augment(augment_factor,val_destructive=False)
    loader= DataLoader(ImagesetFull(storage=data_storage), batch_size=32,shuffle=shuffled)
    return loader