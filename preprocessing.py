import random
from os import error

from PIL import Image
import torch
import albumentations as albu
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
from torchvision import transforms
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from ultralytics import YOLO

def im_labels_pair_getter(folder="21ClassDataset/", label_file="labels_21ClassDataset.csv"):
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

def single_im_preprocessing(image:Image.Image, imsize:int=224, yolocropper=None)->torch.Tensor: # changed size to 224 ~Erik
    """takes a single PIL image and scales it to imsize*imsize pixels(default: 224x224) and  blurs it with a gaussian kernel
    Args:
        image(PIL.Image.Image): the PIL Image to be preprocessed
        imsize(int): the sidelength to which the inputted image should be scaled
    Returns:
            the preprocessed Image, dtype=torch.Tensor"""
    image = image.convert("RGB")
    gaus = GaussianBlur(5, 1)
    image = gaus.forward(image)
    imarray = np.array(image)
    if yolocropper== None:
        yolocropper = YOLO("yolo26n.pt",verbose=False)
    results = yolocropper(imarray,verbose=False)
    if len(results[0].boxes) > 0:
        cropped=imarray
        maxsize=0
        for idx, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            xlen=x2-x1
            ylen=y2-y1
            size=xlen*ylen
            if size>maxsize and (results[0].boxes[idx].cls[0]==15 or results[0].boxes[idx].cls[0]==16):#15=cat 16=dog
                maxsize=size
                cropped = imarray[y1:y2, x1:x2]
    else:
        cropped=imarray
    #cropper=albu.CenterCrop(imsize,imsize)
    #cropped=cropper(image=imarray)['image']
    imtensor=torch.tensor(cropped)
    imtensor = imtensor.permute(2,0,1) #[H,W,C]->[C,H,W]
    resize= transforms.Resize((imsize,imsize))
    imtensor = resize(imtensor)
    return imtensor/255

def list_im_preprocessing(images:list[Image.Image], imsize=128)->list[torch.Tensor]:
    """applies single_im_preprocessing over the given list of images, scaling them to imsize*imsize
    Args:
        images(list[PIL.Image.Image]): the PIL Images to be preprocessed
        imsize(int): the sidelength to which the inputted images should be scaled
    Returns:
            the preprocessed Images, dtype=list[torch.Tensor]"""
    result:list[torch.Tensor] = [None] * len(images) #type: ignore[list-item]
    cropper = YOLO("yolo26n.pt", verbose=False)
    for i in tqdm(range(len(images)),desc="preprocessing for the dataset"):
        result[i]=single_im_preprocessing(images[i],imsize,yolocropper=cropper)
    return result

def image_hide_and_seek(image:torch.Tensor, patches_side:int, patches_length:int)->torch.Tensor:
    for i in range(patches_side):
        for j in range(patches_side):
            if random.choice([0,1]) == 0: #0 for not visible 1 for visible
                hdim:int = i*patches_length
                wdim:int = j*patches_length
                image[:,hdim:hdim+patches_length,wdim:wdim+patches_length] = 0
    return image

class PreprocessedPairStorage():
    def __init__(self,imsize:int,data=None,labels=None):
        if data is None and labels is None:
            temppairs = im_labels_pair_getter()
            imspercent = len(temppairs) / 100
            data = list_im_preprocessing([(x[0]) for x in temppairs], imsize)

            labels = [int(x[1].replace('-1', '20')) for x in temppairs]

            self.data:list[torch.Tensor] = shuffle(data, random_state=0)
            self.labels:list[int] = shuffle(labels, random_state=0)
            self.split:int = round(imspercent * 80)
            self.augmented:bool=False
        elif not(data is None or labels is None):
            self.data = data
            self.labels = labels
            self.split=round((len(data)/100)*80)
            self.augmented=augmented
        else:
            raise error("illegal init of PreprocessedPairStorage")


    def augment(self, factor:int, val_destructive:bool=True, patches:int=16, copy:bool=False):
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
            return self
        else:
            patch_side:int = int(patches**0.5)
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
            if copy:
                return PreprocessedPairStorage(self.data[1].size()[1],new_data,new_labels,True)
            else:
                self.data = new_data
                self.labels = new_labels
                self.augmented = True
                return self


class Imageset(torch.utils.data.Dataset):
    def __init__(self,train:bool,storage:PreprocessedPairStorage,augment:bool=True):
        self.train=train
        if train:
            self.data = storage.data[:storage.split]
            self.labels = storage.labels[:storage.split]
            if augment:
                flip = transforms.RandomHorizontalFlip(1)
                self.data = self.data + [flip(h) for h in tqdm(self.data, desc="flipping")]
                self.labels = self.labels + self.labels
                contrastbright = albu.PlasmaBrightnessContrast(p=1,roughness=2)
                rot = albu.SafeRotate(limit=(-10,10),p=1)
                self.data = self. data + [torch.permute(torch.tensor(contrastbright(image=b.permute(1,2,0).numpy())['image']),(2,0,1)) for b  in tqdm(self.data,desc="brightness/contrast augmentation")]
                self.labels = self.labels + self. labels
                self.data = self.data + [torch.permute(torch.tensor(rot(image=b.permute(1, 2, 0).numpy())['image']), (2, 0, 1)) for b in tqdm(self.data, desc="small(up to 10°) rotation augmentation")]
                self.labels = self.labels + self.labels
                self.data=shuffle(self.data,random_state=1)
                self.labels=shuffle(self.labels,random_state=1)
            print("ims=", len(self.data))
        else:
            self.data = storage.data[storage.split:]
            self.labels = storage.labels[storage.split:]

    def __getitem__(self, item):
        if self.train:
            return self.data[item],torch.tensor(self.labels[item])
        else:
            return self.data[item], torch.tensor(self.labels[item])
    def __len__(self):
        return len(self.data)

class ImagesetFull(torch.utils.data.Dataset):
    def __init__(self,storage:PreprocessedPairStorage, augment:bool=True):
        self.data = storage.data
        self.labels = storage.labels
        if augment:
            flip = transforms.RandomHorizontalFlip(1)
            self.data = self.data + [flip(h) for h in tqdm(self.data, desc="flipping")]
            self.labels = self.labels + self.labels
            contrastbright= albu.PlasmaBrightnessContrast(p=1,roughness=2)
            rot = albu.SafeRotate(limit=(-10,10),p=1)
            self.data = self. data + [torch.permute(torch.tensor(contrastbright(image=b.permute(1,2,0).numpy())['image']),(2,0,1)) for b  in tqdm(self.data,desc="brightness/contrast augmentation")]
            self.labels = self.labels + self.labels
            self.data = self.data + [torch.permute(torch.tensor(rot(image=b.permute(1, 2, 0).numpy())['image']), (2, 0, 1)) for b in tqdm(self.data, desc="small(up to 10°) rotation augmentation")]
            self.labels = self.labels + self.labels
            self.data=shuffle(self.data,random_state=1)
            self.labels=shuffle(self.labels,random_state=1)

    def __getitem__(self, item):
        return self.data[item],torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.data)

def get_dataloaders(shuffled:bool=False, image_side_length:int=224, augment_factor:int=0,train_batch_size=8,augment=True):
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
    train_set = DataLoader(Imageset(train=True,storage=data_storage,augment=augment), batch_size=train_batch_size, shuffle=shuffled)
    return train_set,valid_set

def get_augmented_dataloader_from_augmented_storage(shuffle:bool, augmented_storage:PreprocessedPairStorage,batch_size=8):
    return DataLoader(ImagesetFull(storage=augmented_storage), batch_size=8, shuffle=shuffle)

def get_one_dataloader(shuffled:bool=False, image_side_length:int=224, augment_factor:int=0,batch_size=8):
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
    loader = DataLoader(ImagesetFull(storage=data_storage), batch_size=batch_size,shuffle=shuffled)
    return loader

# Added by Erik
def get_validation_dataloader(image_side_length:int=224, batch_size=8):

    data_pairs = im_labels_pair_getter(folder="images/", label_file="labels.csv")
    data = list_im_preprocessing([(x[0]) for x in data_pairs], image_side_length)
    labels = [int(x[1].replace('-1', '20')) for x in data_pairs]

    data_storage = PreprocessedPairStorage(data=data, labels=labels, imsize=image_side_length)
    loader = DataLoader(ImagesetFull(data_storage, augment=False), batch_size=batch_size)
    return loader


#pairs = im_labels_pair_getter()
#for i in range(10):
#    fig, axs=plt.subplots(2,1)
#    axs[0].imshow(single_im_preprocessing(pairs[3000+i][0]).permute(1,2,0))
#    axs[1].imshow(pairs[3000+i][0])
#    plt.show()
