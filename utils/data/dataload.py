import os
import cv2
import numpy as np
import torch.utils.data as data_utils
from tools.proc_Keratoconus_data import obtain_Keratoconus

from utils.config import cfg
from utils.data.augmentation import Augmentation


def process_img(img_path, label, transform=None):
    """
    Process input image from queue and convert to acceptable format to solver

    @param img_path: 
        tiff file path
    @param label:
        image label
    @param transform:
        augmentation
    @return: 
        image: (*resize, 3)
    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if transform:
        img, label = transform(img, label)

    if len(img.shape) == 3:
        img = img.transpose((2, 0, 1))
    else:
        img = img.transpose((0, 3, 1, 2))

    return img.squeeze(), label


class CancerDataset(data_utils.Dataset):

    def __init__(self, img_root, data_list, transform=None):
        """
        Initialize Data Loader with `data_list`

        :param img_root: image files root
        :param cancer_list: path to data_list.txt list

        """

        self.img_root = img_root
        self.transform = transform

        # Read data list file
        with open(data_list, 'r') as handle:
            data_list = [line.split(',') for line in handle.readlines()]

        self.imgs = [t[0].strip() for t in data_list]
        self.labels = [int(t[1].strip()) for t in data_list]

        self.img_list = np.array(self.imgs)
        self.label_list = np.array(self.labels)

        # Shuffle
        idx = np.random.permutation(self.img_list.shape[0])
        self.img_list = self.img_list[idx]
        self.label_list = self.label_list[idx]

    def __getitem__(self, item):
        image_path = os.path.join(self.img_root, self.img_list[item])
        label = self.label_list[item]
        return process_img(
            image_path,
            label,
            transform=self.transform
        )

    def __len__(self):
        return self.img_list.shape[0]


class DeployDataset(data_utils.Dataset):

    def __init__(self, img_root, data_list, transform=None):

        self.transform = transform
        self.img_root = img_root

        # Read data list file
        with open(data_list, 'r') as handle:
            data_list = handle.readlines()
        self.img_list = [t.split(',')[0].strip() for t in data_list]
        self.img_label = [int(t.split(',')[1].strip()) for t in data_list]

    def __getitem__(self, item):

        image_path = os.path.join(self.img_root, self.img_list[item])
        image_label = self.img_label[item]
        # return self.img_list[item], \
        #        process_img(image_path, image_label, transform=self.transform)
        return process_img(image_path, image_label, transform=self.transform), image_path

    def __len__(self):
        return len(self.img_list)


class Keratoconus_Dataset(data_utils.Dataset):

    def __init__(self, img_root, data_list, transform=None, argu=['CUR', 'PAC'], flip=True, crop=7, front=True, back=True):
        """
        Initialize Data Loader with `data_list`

        :param img_root: image files root
        :param cancer_list: path to data_list.txt list

        """

        self.img_root = img_root
        self.transform = transform
        self.argu = argu
        self.flip = flip
        self.crop = crop
        self.front = front
        self.back = back

        # Read data list file
        with open(data_list, 'r') as handle:
            data_list = [line.split(',') for line in handle.readlines()]

        self.imgs = [t[0].strip() for t in data_list]
        self.labels = [int(t[1].strip()) for t in data_list]

        self.img_list = np.array(self.imgs)
        self.label_list = np.array(self.labels)

        # Shuffle
        idx = np.random.permutation(self.img_list.shape[0])
        self.img_list = self.img_list[idx]
        self.label_list = self.label_list[idx]

    def __getitem__(self, item):
        image_path = os.path.join(self.img_root, self.img_list[item])
        label = self.label_list[item]

        array = obtain_Keratoconus(image_path, argu=self.argu, front=self.front, back=self.back)
        up = list(array[0, :, 0]).index(self.crop)
        dn = list(array[0, :, 0]).index(-self.crop)
        lt = list(array[0, 0, :]).index(-self.crop)
        rt = list(array[0, 0, :]).index(self.crop)
        #array = (array[:, up:dn+1, lt:rt+1]).astype('uint8').transpose((1, 2, 0))
        array = (array[:, up:dn + 1, lt:rt + 1]).transpose((1, 2, 0))
        if self.flip and self.img_list[item][-2:] == 'OS':
            array = array[:, ::-1, :]



        if self.transform:
            array, label = self.transform(array, label)

        if len(array.shape) == 3:
            array = array.transpose((2, 0, 1))
        else:
            array = array.transpose((0, 3, 1, 2))

        return array, label

    def __len__(self):
        return self.img_list.shape[0]


class fuyang_Dataset(data_utils.Dataset):

    def __init__(self, data1, labdata2, count=1000, key='train'):
        """
        Initialize Data Loader with `data_list`

        :param img_root: image files root
        :param cancer_list: path to data_list.txt list

        """
        import pandas as pd
        dfY = pd.read_csv(labdata2, chunksize=count)
        for chunk in dfY:
            Y = chunk.values[:, 1:]
            break
        if key=='test':
            Y = pd.read_csv(labdata2).values[:, 1:][-count:]
        Y = np.array(Y)
        dfX = pd.read_csv(data1, chunksize=count)
        for chunk in dfX:
            X = chunk.values[:, 1:]
            break
        if key=='test':
            X = pd.read_csv(data1).values[:, 1:][-count:]
        X = np.array(X)

        # Shuffle
        if key=='test':
            self.img_list = X
            self.label_list = Y
        else:
            idx = np.random.permutation(Y.shape[0])
            self.img_list = X[idx]
            self.label_list = Y[idx]

    def __getitem__(self, item):
        array = self.img_list[item]
        label = self.label_list[item]

        return array, label

    def __len__(self):
        return self.img_list.shape[0]



