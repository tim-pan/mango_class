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
preprocess = {
    'train': transforms.Compose([
        transforms.RandomRotation(degrees = (0,359)),
        transforms.Resize(280),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees = (0,359)),
        transforms.ColorJitter(brightness=(0, 3), contrast=(
        0, 3), saturation=(0, 5), hue=(0, 0)),
        # transforms.ColorJitter(brightness=(0, 3), contrast=(
        # 0, 5), saturation=(0, 10), hue=(0, 0)),
        # value设置成字符串的进行，就会随机填充。
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.02), ratio=(1, 1), value='1234'),

    ]),
    'dev': transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# preprocess = transforms.Compose([
    # transforms.Resize(100),
    # transforms.CenterCrop(64),
    # transforms.RandomHorizontalFlip(),  
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(degrees = (0,359)),  
#     # transforms.ToTensor(),t
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

     
#     # transforms.ToPILImage(),
#     # transforms.Normalize([0.5], [0.5])
# ])
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
        self.labels = None
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
            # print(fn)
            k = self.train_mango_csv[self.train_mango_csv['image_id']==fn].index.tolist()[0]
            
            self.filenames.append((fn, self.train_mango_csv['label'][k])) # (filename, label) pair
                
        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        self.labels = []
        self.images = []
        #把所有檔案名字對到label的東西取出來preload
        for image_fn, label in self.filenames:            
            # load images
            image = Image.open('.'+self.pic_root+"/"+image_fn)
#             image = mpimg.imread(self.pic_root+"/"+image_fn)
#             print(type(image))
            
            # avoid too many opened files bug
            #抓出來複製，然後再丟進images,最後再關掉
            image = image.copy()  
            # print(image_copy)
            # image_copy = np.array(image_copy, dtype = np.float32) 
            image = np.array(image) 

            # print(image_copy)
            image = Image.fromarray(image, 'RGB')
            # print(image_copy)
#             input_tensor = image_copy
            
#             input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            self.images.append(image)
            #preprocessing_end
            # image.close()
            if label == 'A':
                label = 0
            elif label == 'B':
                label = 1
            elif label == 'C':
                label = 2
            self.labels.append(label)


    def __getitem__(self, index):
        #index是指他在用next時會不斷迭代，從0開始不段增加
    #可以先看一下最最最下面的例子
#https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-6-%E8%BF%AD%E4%BB%A3%E9%82%A3%E4%BB%B6%E5%B0%8F%E4%BA%8B-%E6%B7%B1%E5%85%A5%E4%BA%86%E8%A7%A3-iteration-iterable-iterator-iter-getitem-next-fac5b4542cf4
        """ Get a sample from the dataset
        """
        if self.images != []:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.phase == "train":
            #preprocessing
            image = preprocess["train"](image)

        elif self.phase == "dev" or "val" or "test":

            image = preprocess["dev"](image)

        else :

            print("you had wrong 'phase'")
            
            # label = label.ToTensor()

#             image = self.transform(image)
        # return image and label
        return image, label#這個在下面要visualize會用到


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

class ChaVector(Dataset):
    def __init__(self,
                 preload = False,
                 input_tensor = None,
                 input_label = None):
        """ Intialize the dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        super().__init__()
        self.tensors = []
        self.labels = []
        self.input_tensor = input_tensor
        self.input_label = input_label
        self.len = self.input_tensor.size()[0]

        if preload:
            self._preload()
            
        
                              
    def _preload(self):
        self.labels = []
        self.tensors = []
        #把所有檔案名字對到label的東西取出來preload
        for ind in range(self.len):            
            self.tensors.append(self.input_tensor[ind])
            self.labels.append(self.input_label[ind])


    def __getitem__(self, index):
        #index是指他在用next時會不斷迭代，從0開始不段增加
    #可以先看一下最最最下面的例子
#https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-6-%E8%BF%AD%E4%BB%A3%E9%82%A3%E4%BB%B6%E5%B0%8F%E4%BA%8B-%E6%B7%B1%E5%85%A5%E4%BA%86%E8%A7%A3-iteration-iterable-iterator-iter-getitem-next-fac5b4542cf4
        """ Get a sample from the dataset
        """
        if self.tensors != []:
            # If dataset is preloaded
            tensor = self.tensors[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            tensor, label = self.input_tensor[index], self.input_label[index]
           
        # May use transform function to transform samples
        # e.g., random crop, whitening
        
        return tensor, label#這個在下面要visualize會用到


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
