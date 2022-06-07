import os
import pathlib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
import torchvision.transforms as T
import PIL


class Dataset(data.Dataset):
    def __init__(self, dataset, k_shot=None, train='train'):
        super(Dataset, self).__init__()
        self.k_shot = k_shot
        self.train = train
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
            self.labels = list(np.unique(labels))
            labels_idx = list(map(lambda x: self.labels.index(x), labels))
            df = pd.DataFrame({'img_dir':img_dir, 'labels':labels_idx})
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.labels, random_state=2022)
            if self.train:
                self.df = train_df
            else:
                self.df = test_df

        # SUN397
        elif dataset == 'sun397':
            img_path = './data/sun397/img'
            if self.train == 'train':
                partition_path = './data/sun397/Partitions/Training_01.txt'
            else:
                partition_path = './data/sun397/Partitions/Testing_01.txt'
            img_dir = list(map(lambda x : img_path + x, pathlib.Path(partition_path).read_text().split('\n')))
            labels = list(map(lambda x: x.split('/')[-2], img_dir))
            self.labels = list(np.unique(labels))
            labels_idx = list(map(lambda x: self.labels.index(x), labels))
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels_idx}).iloc[:-1, :]

        # FGVCAircraft
        elif dataset == 'fgvcaircraft':
            img_path = './data/fgvcaircraft/img'
            if self.train == 'train':
                fam_txt = './data/fgvcaircraft/images_manufacturer_train.txt'
                var_txt = './data/fgvcaircraft/images_variant_train.txt'
            elif self.train == 'val':
                fam_txt = './data/fgvcaircraft/images_manufacturer_val.txt'
                var_txt = './data/fgvcaircraft/images_variant_val.txt'
            else:
                fam_txt = './data/fgvcaircraft/images_manufacturer_test.txt'
                var_txt = './data/fgvcaircraft/images_variant_test.txt'
            img_dir = list(map(lambda x: os.path.join(img_path, x)+'.jpg', list(map(lambda x: x.split(' ')[0], pathlib.Path(fam_txt).read_text().split('\n')[:-1]))))
            fam_dir = list(map(lambda x: x.split(' ')[1], pathlib.Path(fam_txt).read_text().split('\n')[:-1]))
            var_dir = list(map(lambda x: x.split(' ')[1], pathlib.Path(var_txt).read_text().split('\n')[:-1]))
            labels = list(map(lambda x, y: x+ ' '+y, fam_dir, var_dir))
            self.labels = list(np.unique(labels))
            labels_idx = list(map(lambda x: self.labels.index(x), labels))
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels_idx})
                
        # subsampling
        if self.train=='train':
            self.df = self.df.groupby('labels').sample(n=self.k_shot, random_state=2022)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = PIL.Image.open(self.df.img_dir.iloc[index])
        if self.train == 'train':
            return T.ToTensor()(img), self.df.labels.iloc[index]
        else:
            return T.ToTensor()(img)