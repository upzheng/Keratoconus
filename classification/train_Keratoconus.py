import torch.utils.data as data_utils
import os
import sys
import torch
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from classification.solver import ImageSolver
from utils.data.dataload import CancerDataset, Keratoconus_Dataset
from utils.data.augmentation import BaseAugmentation, Augmentation, Kerat_Augmentation
from options.base_train_option import BaseTrainOptions_Ker, str2bool


class KeratOption(BaseTrainOptions_Ker):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--resnet_layers', default=18, type=int, choices=[18, 34, 50], help='Number of Resnet layers')
        self.parser.add_argument('--resnet_dropout', default=0.5, type=float, help='Dropout rate of Resnet')
        self.parser.add_argument('--Attrib', default=['CUR', 'PAC'], nargs='+', help='Attribute for people')
        self.parser.add_argument('--Attrib_F', default=True, type=str2bool, help='Whether to use Front')
        self.parser.add_argument('--Attrib_B', default=True, type=str2bool, help='Whether to use Back')
        self.parser.add_argument('--input_dim', default=3, type=int, help='input dimension of image')
        self.parser.add_argument('--OS_mirror', default=True, type=str2bool, help='Whether to mirror os')
        self.parser.add_argument('--crop', default=7, type=float, help='crop radius')
        self.parser.add_argument('--conc_fuse', default=False, type=str2bool, help='Whether to concate feature in model')


def main(args):
    seed = 666
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data = Keratoconus_Dataset(
        img_root=args.img_root,
        data_list=args.train_csv,
        transform=Kerat_Augmentation(size=args.input_size, rescale=args.rescale, mean=args.means, std=args.stds),
        argu=args.Attrib,
        flip=args.OS_mirror,
        crop=args.crop,
        front=args.Attrib_F,
        back=args.Attrib_B
    )

    train_load = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    val_data = Keratoconus_Dataset(
        img_root=args.img_root,
        data_list=args.val_csv,
        transform=Kerat_Augmentation(size=args.input_size, rescale=args.rescale, mean=args.means, std=args.stds, mode='val'),
        argu=args.Attrib,
        flip=args.OS_mirror,
        crop=args.crop,
        front=args.Attrib_F,
        back=args.Attrib_B
    )
    val_load = data_utils.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    solver = ImageSolver(args)
    solver.train(train_load, val_load)


if __name__ == '__main__':
    options = KeratOption()
    args = options.initialize()
    main(args)
