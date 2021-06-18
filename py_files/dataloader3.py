import os
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image#https://yungyuc.github.io/oldtech/python/python_imaging.html
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dtype = torch.float32 # we will be using float throughout this tutorial

#the steps of preprocessing


trans_list = [
        transforms.RandomRotation(degrees = (0,359)),#
        transforms.Resize(280),
        transforms.CenterCrop(224),

        transforms.RandomHorizontalFlip(0.5), #3 
        transforms.RandomVerticalFlip(0.5),#4

        transforms.ColorJitter(brightness=(0, 3), contrast=(
        0, 3), saturation=(0, 5), hue=(0, 0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.02), ratio=(1, 1), value='1234'),
    
]
preprocess = {
    'train': transforms.Compose(trans_list),
    'dev': transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class mango(Dataset):
    """
    A customized data loader for mango.
    """
    def __init__(self,
                 pic_root,
                 label_root,
                 transform=None,
                 preload=False,
                 phase=None):
        """ Intialize the dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        super().__init__()
        self.images = None
        self.sample_pic_path = None
        self.train_mango_files = None
        self.train_mango_csv = None
        self.pic_name = None
        self.filenames = []
        self.pic_root = pic_root
        self.label_root = label_root
        self.phase = phase

        #https://www.itread01.com/articles/1476166832.html
        self.sample_pic_path = os.path.join("."+self.pic_root)
        self.train_mango_files = os.listdir(self.sample_pic_path)
        
        self.train_mango_csv = pd.read_csv("."+self.label_root, encoding='utf-8')
        self.pic_name=self.train_mango_csv['image_id']#series
        self.pic_name=list(self.pic_name)

        # print(len(self.train_mango_files))#800
        
        for fn in self.train_mango_files:
            if 'jpg' in fn:

                k = self.train_mango_csv[self.train_mango_csv['image_id']==fn].index.tolist()[0]
                
                self.filenames.append((fn, self.train_mango_csv['label'][k])) # (filename, label) pair
                
        # if preload dataset into memory
        if preload:
            self._preload()

        if self.phase == "train":
            self.len = len(self.filenames) * 2
        elif self.phase == "dev":
            self.len = len(self.filenames)
            
        
                              
    def _preload(self):
        self.images = []
        #把所有檔案名字對到label的東西取出來preload
        for image_fn, label in self.filenames:            
            # load images
            if label == 'A':
                label = 0
            elif label == 'B':
                label = 1
            elif label == 'C':
                label = 2

            image = Image.open('.'+self.pic_root+"/"+image_fn)

#             image = mpimg.imread(self.pic_root+"/"+image_fn)
#             print(type(image))
            
            # avoid too many opened files bug
            #抓出來複製，然後再丟進images,最後再關掉
            image = image.copy() 
            if self.phase == "train":
            #preprocessing1
                trans_list[3] = transforms.RandomHorizontalFlip(0) #3 
                trans_list[4] = transforms.RandomVerticalFlip(0)#4
                preprocess["train"] = transforms.Compose(trans_list)
                image1 = preprocess["train"](image)
                # print(0, end = '')
            #preprocessing2 
                trans_list[3] = transforms.RandomHorizontalFlip(1) #3 
                trans_list[4] = transforms.RandomVerticalFlip(1)#4
                preprocess["train"] = transforms.Compose(trans_list)
                image2 = preprocess["train"](image)
                # print(1, end = '')
    

                self.images.append([image1, label])
                self.images.append([image2, label])


                    # print(2, end = '')

            elif self.phase == "dev" or "val" or "test":
                image = preprocess["dev"](image)
                self.images.append([image, label])

            else :

                print("you had wrong 'phase'")

        
    def __getitem__(self, index):
        #index是指他在用next時會不斷迭代，從0開始不段增加
    #可以先看一下最最最下面的例子
#https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-6-%E8%BF%AD%E4%BB%A3%E9%82%A3%E4%BB%B6%E5%B0%8F%E4%BA%8B-%E6%B7%B1%E5%85%A5%E4%BA%86%E8%A7%A3-iteration-iterable-iterator-iter-getitem-next-fac5b4542cf4
        """ Get a sample from the dataset
        """
        if self.images != []:
            # If dataset is preloaded
            image, label = self.images[index]
            
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        return image, label
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
