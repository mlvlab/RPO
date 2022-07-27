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
import json



# for base class training / evaluation (base class or novel class)
class UnseenDataset(data.Dataset):
    def __init__(self, dataset, k_shot=None, train='train', base_label_ratio=0.5, train_time=None, test_time=None, device=torch.device('cpu')):
        '''
        train_time : one of ['entire', 'base']
        test_time : one of ['entire', 'base', 'novel']
        '''
        super(UnseenDataset, self).__init__()
        self.k_shot = k_shot
        self.base_label_ratio = base_label_ratio
        self.train = train
        self.train_time = train_time
        self.test_time = test_time
        self.transform = T.Compose([T.Resize((299, 299))])
        self.device = device
        # create img_dir, text label list
        img_dir = []
        labels = []

        # EuroSAT
        if dataset == 'eurosat':
            img_path = './data/eurosat/'
            with open('./data/eurosat/split_zhou_EuroSAT.json') as f:
                split = json.load(f)
            
            if self.train == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels'])
            elif self.train == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels'])
            self.labels = list(np.unique(np.array(split["train"])[:,2]))

        # SUN397
        elif dataset == 'sun397':
            img_path = './data/sun397/img'
            if self.train == 'train':
                partition_path = './data/sun397/Partitions/Training_01.txt'
            else:
                partition_path = './data/sun397/Partitions/Testing_01.txt'
            img_dir = list(map(lambda x : img_path + x, pathlib.Path(partition_path).read_text().split('\n')))
            _ = img_dir.pop(-1)
            labels = list(map(lambda x: x.split('/')[5], img_dir))
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
            self.labels_li = list(map(lambda x, y: x+ ' '+y, fam_dir, var_dir))
            self.labels = list(np.unique(self.labels_li))
            labels_idx = list(map(lambda x: self.labels.index(x), self.labels_li))
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels_idx})
                
        # subsampling
        if self.train=='train':
            self.df = self.df.groupby('labels').sample(n=self.k_shot, random_state=2022)
        if self.train == 'val':
            self.df = self.df.groupby('labels').sample(n=min(self.k_shot, 4), random_state=2022)

        # load image to cuda at once
        self.df['image'] = self.df['img_dir'].map(self.dir_to_tensor)
        # divide base classes and novel classes
        self.divide()
    
    def dir_to_tensor(self, dir):
        img = PIL.Image.open(dir)
        img = T.ToTensor()(img)
        if img.shape[0] == 1:
            img = torch.cat([img, img, img], dim=0)
        elif img.shape[0] == 4:
            img = img[:3,:,:]
        img = self.transform(img).to(self.device)
        return img

    def divide(self):
        # assign base_label_ratio of class label as base classes
        self.base_df = self.df[self.df.labels < (self.df.labels.max()+1)*self.base_label_ratio].copy()
        label_idx = list(np.unique(self.base_df.labels.values))
        self.base_labels = np.array(self.labels)[label_idx].tolist()
        self.base_df['labels'] -= self.base_df['labels'].min()
        # assign rest as novel classes
        self.novel_df = self.df[self.df.labels >= (self.df.labels.max()+1)*self.base_label_ratio].copy()
        label_idx = list(np.unique(self.novel_df.labels.values))
        self.novel_labels = np.array(self.labels)[label_idx].tolist()
        self.novel_df['labels'] -= self.novel_df['labels'].min()

    def __len__(self):
        if self.train == 'train':
            if self.train_time == 'entire':
                return len(self.df)
            elif self.train_time == 'base':
                return len(self.base_df)
        elif self.train == 'test':
            if self.test_time == 'entire':
                return len(self.df)
            elif self.test_time == 'novel':
                return len(self.novel_df)
            elif self.test_time == 'base':
                return len(self.base_df)

    def __getitem__(self, index):
        if self.train == 'train':
            if self.train_time == 'entire':
                return self.df.image.iloc[index], self.df.labels.iloc[index]
            elif self.train_time == 'base':
                return self.base_df.image.iloc[index], self.base_df.labels.iloc[index]
        
        elif self.train == 'test':
            if self.test_time == 'entire':
                return self.df.image.iloc[index]
            elif self.test_time == 'base':
                return self.base_df.image.iloc[index]
            elif self.test_time == 'novel':
                return self.novel_df.image.iloc[index]