import os
import pathlib
import math
import random
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
import torchvision
import torchvision.transforms as T
import PIL
from PIL import Image
import json

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC




# for base class training / evaluation (base class or novel class)
class UnseenDataset(data.Dataset):
    def __init__(self, dataset, k_shot=None, mode='train', base_label_ratio=0.5, train_time=None, test_time=None, device=torch.device('cpu')):
        '''
        train_time : one of ['entire', 'base']
        test_time : one of ['entire', 'base', 'novel']
        mode : one of ['constrained', 'generalized']
        '''
        super(UnseenDataset, self).__init__()
        self.k_shot = k_shot
        self.base_label_ratio = base_label_ratio
        self.mode = mode
        self.train_time = train_time
        self.test_time = test_time
        self.device = device

        self.train_transform = T.Compose([
                                     T.Resize((224,224),  interpolation=BICUBIC),
                                     T.RandomHorizontalFlip(p=0.5),
                                     T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                                    ])
        self.test_transform = T.Compose([
                                     T.Resize((224,224), interpolation=BICUBIC),
                                     T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                                    ])
        
        if dataset == 'imagenet':

            if self.mode == 'train':
                root = '/hub_data1/imagenet/train/' 
                img_dir, labels, name = [], [], []
                # map label_id and label name
                with open('/home/dongjun/VC_prompt_learning/data/imagenet/classnames.txt') as f:
                    annot = f.readlines()
                label_map = {}
                for label in annot:
                    label = label.replace('\n','')
                    id, label_name = label[:9], label[10:]
                    label_map[id] = label_name
                # create dataframe
                for idx, label_id in enumerate(os.listdir(root)):
                    label_name = label_map[label_id]
                    for img in os.listdir(os.path.join(root, label_id)):
                        temp = os.path.join(root, label_id, img)
                        img_dir.append(temp)
                        labels.append(idx)
                        name.append(label_name)
                self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
                self.labels = list(map(lambda x: x[0], list(self.df.groupby('labels')['name'].unique())))
            elif self.mode == 'test':
                root = '/hub_data1/imagenet/val/' 
                img_dir, labels, name = [], [], []
                # map label_id and label name
                with open('/home/dongjun/VC_prompt_learning/data/imagenet/classnames.txt') as f:
                    annot = f.readlines()
                label_map = {}
                for label in annot:
                    label = label.replace('\n','')
                    id, label_name = label[:9], label[10:]
                    label_map[id] = label_name
                # create dataframe
                for idx, label_id in enumerate(os.listdir(root)):
                    label_name = label_map[label_id]
                    for img in os.listdir(os.path.join(root, label_id)):
                        temp = os.path.join(root, label_id, img)
                        img_dir.append(temp)
                        labels.append(idx)
                        name.append(label_name)
                self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
                self.labels = list(map(lambda x: x[0], list(self.df.groupby('labels')['name'].unique())))
            else:
                pass
        
        elif dataset == 'imagenet_sketch':
            root = './data/imagenet_sketch/'
            img_dir, labels, name = [], [], []
            # map label_id and label name
            with open('./data/imagenet/classnames.txt') as f:
                annot = f.readlines()
            label_map = {}
            for label in annot:
                label = label.replace('\n','')
                id, label_name = label[:9], label[10:]
                label_map[id] = label_name
            # create dataframe
            for idx, label_id in enumerate(os.listdir(root)):
                label_name = label_map[label_id]
                for img in os.listdir(os.path.join(root, label_id)):
                    temp = os.path.join(root, label_id, img)
                    img_dir.append(temp)
                    labels.append(idx)
                    name.append(label_name)
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
            self.labels = list(map(lambda x: x[0], list(self.df.groupby('labels')['name'].unique())))

        elif dataset == 'imagenet_a':
            root = './data/imagenet_a/'
            img_dir, labels, name = [], [], []
            # map label_id and label name
            with open('./data/imagenet/classnames.txt') as f:
                annot = f.readlines()
            label_map = {}
            for label in annot:
                label = label.replace('\n','')
                id, label_name = label[:9], label[10:]
                label_map[id] = label_name
            # create dataframe
            for idx, label_id in enumerate(os.listdir(root)):
                label_name = label_map[label_id]
                for img in os.listdir(os.path.join(root, label_id)):
                    temp = os.path.join(root, label_id, img)
                    img_dir.append(temp)
                    labels.append(idx)
                    name.append(label_name)
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
            self.labels = list(map(lambda x: x[0], list(self.df.groupby('labels')['name'].unique())))

        elif dataset == 'imagenet_r':
            root = './data/imagenet_r/'
            img_dir, labels, name = [], [], []
            # map label_id and label ㅎ
            with open('./data/imagenet/classnames.txt') as f:
                annot = f.readlines()
            label_map = {}
            for label in annot:
                label = label.replace('\n','')
                id, label_name = label[:9], label[10:]
                label_map[id] = label_name
            # create dataframe
            for idx, label_id in enumerate(os.listdir(root)):
                label_name = label_map[label_id]
                for img in os.listdir(os.path.join(root, label_id)):
                    temp = os.path.join(root, label_id, img)
                    img_dir.append(temp)
                    labels.append(idx)
                    name.append(label_name)
            self.df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
            self.labels = list(map(lambda x: x[0], list(self.df.groupby('labels')['name'].unique())))

        # EuroSAT
        elif dataset == 'eurosat':
            img_path = './data/eurosat/'
            with open('./data/eurosat/split_zhou_EuroSAT.json') as f:
                split = json.load(f)
            
            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)

        # SUN397
        elif dataset == 'sun397':
            img_path = './data/sun397/img'
            with open('./data/sun397/split_zhou_SUN397.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)
            
        # Caltech101
        elif dataset == 'caltech101':
            img_path = './data/caltech101/img'
            with open('./data/caltech101/split_zhou_Caltech101.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)
        
        # Food101
        elif dataset == 'food101':
            img_path = './data/food101/img'
            with open('./data/food101/split_zhou_Food101.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)

        # Food101
        elif dataset == 'oxfordpet':
            img_path = './data/oxfordpet/img'
            with open('./data/oxfordpet/split_zhou_OxfordPets.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)

        # stanford cars
        elif dataset == 'stanfordcars':
            img_path = './data/stanfordcars/img'
            with open('./data/stanfordcars/split_zhou_StanfordCars.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)

        # Flowers102
        elif dataset == 'flowers102':
            img_path = './data/flowers102/img'
            with open('./data/flowers102/split_zhou_OxfordFlowers.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)

        # DTD
        elif dataset == 'dtd':
            img_path = './data/dtd/img'
            with open('./data/dtd/split_zhou_DescribableTextures.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)

        # UCF101
        elif dataset == 'ucf101':
            img_path = './data/ucf101/img'
            with open('./data/ucf101/split_zhou_UCF101.json') as f:
                split = json.load(f)

            if self.mode == 'train':
                train = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["train"])
                self.df = pd.DataFrame(train, columns = ['img_dir', 'labels', 'name'])
            elif self.mode == 'val':
                val = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["val"])
                self.df = pd.DataFrame(val, columns = ['img_dir', 'labels', 'name'])
            else:
                test = map(lambda x: ["/".join([img_path, x[0]]), x[1], x[2]], split["test"])
                self.df = pd.DataFrame(test, columns = ['img_dir', 'labels', 'name'])
            self.labels = []
            labels = list(self.df.sort_values('labels').name)
            for i in labels:
                if i not in self.labels:
                    self.labels.append(i)


        # FGVCAircraft
        elif dataset == 'fgvcaircraft':
            img_path = './data/fgvcaircraft/img'
            if self.mode == 'train':
                fam_txt = './data/fgvcaircraft/images_manufacturer_train.txt'
                var_txt = './data/fgvcaircraft/images_variant_train.txt'
            elif self.mode == 'val':
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
        if self.mode=='train':
            self.generate_few_shot_df()
            print(self.df.shape[0])
            #self.df = self.df.groupby('labels').sample(n=self.k_shot, random_state=2022)
        if self.mode == 'val':
            self.df = self.df.groupby('labels').sample(n=min(self.k_shot, 4), random_state=2022)

        # divide base classes and novel classes
        self.divide()
        
        # load image to cuda at once(when few-shot training)
        if self.mode == 'train':
            if self.train_time == 'base':
                self.base_df['image'] = self.base_df['img_dir'].map(self.dir_to_tensor)
            elif self.train_time == 'novel':
                self.novel_df['image'] = self.novel_df['img_dir'].map(self.dir_to_tensor)
            else:
                self.df['image'] = self.df['img_dir'].map(self.dir_to_tensor)
    
    def generate_few_shot_df(self):
        sampled_items = []
        labels = []
        names = []
        for label, name in enumerate(self.labels):
            items = list(self.df[self.df.name == name].img_dir.values)
            sampled = random.sample(items, self.k_shot)
            sampled_items += sampled
            labels += [label] * self.k_shot
            names += [name] * self.k_shot
        self.df = pd.DataFrame({'img_dir':sampled_items, 'labels':labels, 'name':names})

    def dir_to_tensor(self, dir):
        img = PIL.Image.open(dir)
        img = T.ToTensor()(img)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], dim=0)
        elif img.shape[0] == 4:
            img = img[:3,:,:]

        if self.mode == 'train':
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)

        img = img.to(self.device)
        return img

    def divide(self):
        # lowercase the label name
        #self.labels = list(map(lambda x : x.lower(), self.labels))
        #self.df.name = self.df.name.str.lower()
        # sort the label name and relabeling
        self.labels.sort()
        n = len(self.labels)
        m = math.ceil(n / 2)
        self.base_labels = self.labels[:m]
        self.novel_labels = self.labels[m:]
        relabeler = {y:y_new for y_new, y in enumerate(self.labels)}
        self.df['labels'] = self.df['name'].map(relabeler)
        self.base_df = self.df[self.df.name.isin(self.base_labels)]
        self.novel_df = self.df[self.df.name.isin(self.novel_labels)]

    def __len__(self):
        if self.mode == 'train':
            if self.train_time == 'entire':
                return len(self.df)
            elif self.train_time == 'base':
                return len(self.base_df)
        elif self.mode == 'test':
            if self.test_time == 'entire':
                return len(self.df)
            elif self.test_time == 'novel':
                return len(self.novel_df)
            elif self.test_time == 'base':
                return len(self.base_df)

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.train_time == 'entire':
                #dir = self.df.img_dir.iloc[index]
                #img = self.dir_to_tensor(dir)
                img = self.df.image.iloc[index]
                return img, self.df.labels.iloc[index]
            elif self.train_time == 'base':
                #dir = self.base_df.img_dir.iloc[index]
                #img = self.dir_to_tensor(dir)
                img = self.base_df.image.iloc[index]
                return img, self.base_df.labels.iloc[index]
        
        elif self.mode == 'test':
            if self.test_time == 'entire':
                dir = self.df.img_dir.iloc[index]
                img = self.dir_to_tensor(dir)
                return img
            elif self.test_time == 'base':
                dir = self.base_df.img_dir.iloc[index]
                img = self.dir_to_tensor(dir)
                return img
            elif self.test_time == 'novel':
                dir = self.novel_df.img_dir.iloc[index]
                img = self.dir_to_tensor(dir)
                return img



class CILImageNet100(data.Dataset):
    def __init__(self, mode='train', cur_step=0, n_tasks=10, device=torch.device('cpu')):
        super(CILImageNet100, self).__init__()
        self.mode = mode
        self.device = device
        self.train_transform = T.Compose([
                                     T.Resize((224,224),  interpolation=BICUBIC),
                                     T.RandomHorizontalFlip(p=0.5),
                                     T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                                    ])
        self.test_transform = T.Compose([
                                     T.Resize((224,224), interpolation=BICUBIC),
                                     T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                                    ])

        if self.mode == 'train':
            root = '/hub_data1/imagenet100/train/' 
            img_dir, labels, name = [], [], []
            # map label_id and label name
            with open('/home/dongjun/VC_prompt_learning/data/imagenet/classnames.txt') as f:
                annot = f.readlines()
            label_map = {}
            for label in annot:
                label = label.replace('\n','')
                id, label_name = label[:9], label[10:]
                label_map[id] = label_name
            # create dataframe
            for idx, label_id in enumerate(os.listdir(root)):
                label_name = label_map[label_id]
                for img in os.listdir(os.path.join(root, label_id)):
                    temp = os.path.join(root, label_id, img)
                    img_dir.append(temp)
                    labels.append(idx)
                    name.append(label_name)
            df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
            self.labels = list(map(lambda x: x[0], list(df.groupby('labels')['name'].unique())))
            self.n_cls = len(self.labels)
            self.task_df = df[df.name.isin(self.labels[int(cur_step*(self.n_cls/n_tasks)):int((cur_step+1)*(self.n_cls/n_tasks))])]
            self.task_df.labels = self.task_df.labels - self.task_df.labels.min()

        elif self.mode == 'test':
            root = '/hub_data1/imagenet100/val/' 
            img_dir, labels, name = [], [], []
            # map label_id and label name
            with open('/home/dongjun/VC_prompt_learning/data/imagenet/classnames.txt') as f:
                annot = f.readlines()
            label_map = {}
            for label in annot:
                label = label.replace('\n','')
                id, label_name = label[:9], label[10:]
                label_map[id] = label_name
            # create dataframe
            for idx, label_id in enumerate(os.listdir(root)):
                label_name = label_map[label_id]
                for img in os.listdir(os.path.join(root, label_id)):
                    temp = os.path.join(root, label_id, img)
                    img_dir.append(temp)
                    labels.append(idx)
                    name.append(label_name)
            df = pd.DataFrame({'img_dir':img_dir, 'labels':labels, 'name':name})
            self.labels = list(map(lambda x: x[0], list(self.df.groupby('labels')['name'].unique())))
            self.n_cls = len(self.labels)
            self.task_df = df[df.name.isin(self.labels[int(cur_step*(self.n_cls/n_tasks)):int((cur_step+1)*(self.n_cls/n_tasks))])]
            self.task_df.labels = self.task_df.labels - self.task_df.labels.min()
        else:
            pass

    def dir_to_tensor(self, dir):
        img = PIL.Image.open(dir)
        img = T.ToTensor()(img)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], dim=0)
        elif img.shape[0] == 4:
            img = img[:3,:,:]

        if self.mode == 'train':
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)

        img = img.to(self.device)
        return img

    def __len__(self):
        return len(self.task_df)
    
    def __getitem__(self, index):
        dir = self.task_df.img_dir.iloc[index]
        img = self.dir_to_tensor(dir)
        return img, self.task_df.labels.iloc[index]
        
            