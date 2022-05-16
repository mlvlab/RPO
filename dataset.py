import os
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torchvision.transforms as T
import PIL



class Dataset(data.Dataset):
    def __init__(self, dataset, k_shot, train=True):
        super(Dataset, self).__init__()
        self.k_shot = k_shot
        self.train = train
        # transform pipeline
        self.transforms = T.Compose([T.ToTensor(),
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        # create img_dir, text label list
        img_dir = []
        labels = []

        # EuroSAT
        if dataset == 'eurosat':
            img_path = './data/eurosat/train'
            for c in os.listdir(img_path):
                if c == '.DS_Store':
                    pass
                else:
                    img_dir += map(lambda x: os.path.join(img_path, c)+'/'+x, os.listdir(os.path.join(img_path, c)))
            labels = list(map(lambda x: x.split('/')[-2], img_dir))
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels})
            self.labels = list(np.unique(labels))

        # SUN397
        elif dataset == 'sun397':
            img_path = './data/sun397/img/'
            if self.train:
                partition_path = './data/sun397/Partitions/Training_01.txt'
            else:
                partition_path = './data/sun397/Partitions/Testing_01.txt'
            img_dir = list(map(lambda x : os.path.join(img_path, x), pathlib.Path(partition_path).read_text().split('\n')))
            labels = list(map(lambda x: x.split('/')[-2], img_dir))
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels})
            self.labels = list(np.unique(labels))

        if self.train:
            self.subsampling()

    # for few-shot training
    def subsampling(self):
        pass

    def __len__(self):
        return len(self.df)

    def __getitem(self, index):
        img = PIL.Image.open(self.df.img_dir[index])
        x = self.transforms(img)
        if self.train:
            return x, self.df.labels[index]
        else:
            return x


            



