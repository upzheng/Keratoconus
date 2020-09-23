import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseTrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('exp_name', type=str, help='Experiment name')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--gpu_id', default=0, type=str, help='GPU ID Use to train/test model')
        self.parser.add_argument('--save_folder', default='save/', help='Path to save checkpoint models')
        self.parser.add_argument('--save_log', help='Path to save checkpoint models')
        self.parser.add_argument('--img_root', type=str, help='Image root')
        self.parser.add_argument('--train_csv', type=str, help='Path of training label files')
        self.parser.add_argument('--val_csv', type=str, help='Path of validation label files')
        self.parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='Training Loss')
        self.parser.add_argument('--num_classes', default=2, type=int, help='# lesion + bg')
        self.parser.add_argument('--model_name', default='ResNet', type=str,
                                 choices=['KerNet', ], help='model')
        self.parser.add_argument('--debug', default=True, type=str2bool, help='Whether to debug')

        # train opts
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_iters', default=50000, type=int, help='Number of training iterations')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='poly', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[25000, 50000, 75000], type=list, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--save_freq', default=400, type=int, help='save weights every # iterations')
        self.parser.add_argument('--display_freq', default=10, type=int, help='display training metrics every # iterations,  =train_epoch_size/2, train_epoch_size = train_total_num/batch_size') #20 for one step data show
        self.parser.add_argument('--val_freq', default=18, type=int, help='do validation every # iterations, =train_epoch_size-1') #100
        self.parser.add_argument('--patience_num', default=5, type=int, help='Tolerance times of rises of the validation set loss')

        # data args
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--label_weights', type=float, default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), nargs='+', help='weight of labels')
        self.parser.add_argument('--means', type=float, default=(0.485, 0.456, 0.406), nargs='+', help='mean')
        self.parser.add_argument('--stds', type=float, default=(0.229, 0.224, 0.225), nargs='+', help='std')
        self.parser.add_argument('--input_size', default=512, type=int, help='model input size')
        self.parser.add_argument('--input_dim', default=3, type=int, help='model input dimention')
        self.parser.add_argument('--gpu-ids', type=str, default='0',
                                 help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)

        print('--------------Options Log-------------')
        print(arg2str(self.args))
        print('--------------Options End-------------')

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_folder):
            os.mkdir(self.args.save_folder)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        print('model save path:', model_save_path)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args


class BaseTrainOptions_Ker(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('exp_name', type=str, help='Experiment name')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--gpu_id', default=0, type=str, help='GPU ID Use to train/test model')
        self.parser.add_argument('--save_folder', default='save/', help='Path to save checkpoint models')
        self.parser.add_argument('--save_log', help='Path to save checkpoint models')
        self.parser.add_argument('--img_root', type=str, help='Image root')
        self.parser.add_argument('--train_csv', type=str, help='Path of training label files')
        self.parser.add_argument('--val_csv', type=str, help='Path of validation label files')
        self.parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='Training Loss')
        self.parser.add_argument('--num_classes', default=2, type=int, help='# lesion + bg' )
        self.parser.add_argument('--model_name', default='ResNet', type=str,
                                 choices=['KerNet', ], help='model')
        self.parser.add_argument('--debug', default=True, type=str2bool, help='Whether to debug')

        # train opts
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_iters', default=50000, type=int, help='Number of training iterations')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='poly', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[25000, 50000, 75000], type=list, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--save_freq', default=400, type=int, help='save weights every # iterations')
        self.parser.add_argument('--display_freq', default=10, type=int, help='display training metrics every # iterations,  =train_epoch_size/2, train_epoch_size = train_total_num/batch_size') #20 for one step data show
        self.parser.add_argument('--val_freq', default=18, type=int, help='do validation every # iterations, =train_epoch_size-1') #100
        self.parser.add_argument('--patience_num', default=5, type=int, help='Tolerance times of rises of the validation set loss')

        # data args
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--label_weights', type=float, default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), nargs='+', help='weight of labels')
        self.parser.add_argument('--means', type=float, default=(0.485, 0.456, 0.406), nargs='+', help='mean')   # imagenet (0.485, 0.456, 0.406)..tongji(0.83, 0.64, 0.86)
        self.parser.add_argument('--stds', type=float, default=(0.229, 0.224, 0.225), nargs='+', help='std') # imagenet (0.229, 0.224, 0.225),tongji (0.14, 0.28, 0.16)
        self.parser.add_argument('--input_size', default=512, type=int, help='model input size')
        self.parser.add_argument('--gpu-ids', type=str, default='0',
                                 help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
        self.parser.add_argument('--seresnet', default=False, type=str2bool, help='Whether to use se in resnet')

    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)

        print('--------------Options Log-------------')
        print(arg2str(self.args))
        print('--------------Options End-------------')

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_folder):
            os.mkdir(self.args.save_folder)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        print('model save path:', model_save_path)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

