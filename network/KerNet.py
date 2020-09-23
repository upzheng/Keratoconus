import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from network.factory import create_model
from network.basenet import BaseNet
from collections import OrderedDict
from network.atten_se import SpatialAttention

class KerNet(BaseNet):

    def __init__(self, args):

        super().__init__(args)

        self.keep_prob = args.resnet_dropout
        self.layers = args.resnet_layers
        self.seresnet = args.seresnet

        model = getattr(torchvision.models, 'resnet{}'.format(self.layers))

        self.resnet0 = model(pretrained=True)
        self.resnet1 = model(pretrained=True)
        self.resnet2 = model(pretrained=True)
        self.resnet3 = model(pretrained=True)
        self.resnet4 = model(pretrained=True)

        self.SpaAtten12_0 = SpatialAttention(kernel_size=7)
        self.SpaAtten12_1 = SpatialAttention(kernel_size=7)
        self.SpaAtten12_2 = SpatialAttention(kernel_size=7)
        self.SpaAtten12_3 = SpatialAttention(kernel_size=7)
        self.SpaAtten12_4 = SpatialAttention(kernel_size=7)


        layer2_conv1_weight = self.resnet0._modules['layer2']._modules['0']._modules['conv1'].weight.data

        layer2_down_weight = self.resnet0._modules['layer2']._modules['0']._modules['downsample']._modules['0'].weight.data

        new_layer2_conv1_x2 = nn.Conv2d(64*2, 128, kernel_size=3, stride=2, padding=1, bias=False)
        new_layer2_conv1_x2.weight.data = torch.cat([layer2_conv1_weight, layer2_conv1_weight], dim=1).clone()
        new_layer2_conv1_x3 = nn.Conv2d(64*3, 128, kernel_size=3, stride=2, padding=1, bias=False)
        new_layer2_conv1_x3.weight.data = torch.cat([layer2_conv1_weight, layer2_conv1_weight, layer2_conv1_weight], dim=1).clone()

        new_layer2_down_x2 = nn.Conv2d(64*2, 128, kernel_size=1, stride=2, bias=False)
        new_layer2_down_x2.weight.data = torch.cat([layer2_down_weight, layer2_down_weight], dim=1).clone()
        new_layer2_down_x3 = nn.Conv2d(64*3, 128, kernel_size=1, stride=2, bias=False)
        new_layer2_down_x3.weight.data = torch.cat([layer2_down_weight, layer2_down_weight, layer2_down_weight], dim=1).clone()

        self.resnet0.layer2[0].conv1 = new_layer2_conv1_x2
        self.resnet1.layer2[0].conv1 = new_layer2_conv1_x3
        self.resnet2.layer2[0].conv1 = new_layer2_conv1_x3
        self.resnet3.layer2[0].conv1 = new_layer2_conv1_x3
        self.resnet4.layer2[0].conv1 = new_layer2_conv1_x2

        self.resnet0.layer2[0].downsample[0] = new_layer2_down_x2
        self.resnet1.layer2[0].downsample[0] = new_layer2_down_x3
        self.resnet2.layer2[0].downsample[0] = new_layer2_down_x3
        self.resnet3.layer2[0].downsample[0] = new_layer2_down_x3
        self.resnet4.layer2[0].downsample[0] = new_layer2_down_x2

        self.feature0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet0.bn1,
            self.resnet0.relu,
            self.resnet0.maxpool,
            self.resnet0.layer1,
        )

        self.feature1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet1.bn1,
            self.resnet1.relu,
            self.resnet1.maxpool,
            self.resnet1.layer1,
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet2.bn1,
            self.resnet2.relu,
            self.resnet2.maxpool,
            self.resnet2.layer1,
        )

        self.feature3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet3.bn1,
            self.resnet3.relu,
            self.resnet3.maxpool,
            self.resnet3.layer1,
        )

        self.feature4 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet4.bn1,
            self.resnet4.relu,
            self.resnet4.maxpool,
            self.resnet4.layer1,
        )

        self.fc = nn.Linear(512*5, 1000)
        self.top = nn.Linear(1000, args.num_classes)

        if self.keep_prob < 1.0:
            self.dropout = nn.Dropout(p=self.keep_prob)

    def model_name(self):
        return 'Resnet-{}'.format(self.layers)

    # make some changes to the end layer contrast to the original resnet
    def forward(self, x):
        # CUR*2 ELE*2 PAC -> PAC ELE_F ELE_B CUR_B CUR_F
        x0, x1, x2, x3, x4 = x[:, 4:5, ...], x[:, 2:3, ...], x[:, 3:4, ...], x[:, 1:2, ...], x[:, 0:1, ...]

        x0 = self.feature0(x0)
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)
        x3 = self.feature3(x3)
        x4 = self.feature4(x4)

        x0_input = torch.cat([x0, x1], dim=1)
        x1_input = torch.cat([x0, x1, x2], dim=1)
        x2_input = torch.cat([x1, x2, x3], dim=1)
        x3_input = torch.cat([x2, x3, x4], dim=1)
        x4_input = torch.cat([x3, x4], dim=1)

        x0_input = x0_input * self.SpaAtten12_0(x0_input)
        x1_input = x1_input * self.SpaAtten12_1(x1_input)
        x2_input = x2_input * self.SpaAtten12_2(x2_input)
        x3_input = x3_input * self.SpaAtten12_3(x3_input)
        x4_input = x4_input * self.SpaAtten12_4(x4_input)

        x0_ = self.resnet0.layer2(x0_input)
        x1_ = self.resnet1.layer2(x1_input)
        x2_ = self.resnet2.layer2(x2_input)
        x3_ = self.resnet3.layer2(x3_input)
        x4_ = self.resnet4.layer2(x4_input)

        x0 = self.resnet0.layer3(x0_)
        x0 = self.resnet0.layer4(x0)
        x1 = self.resnet1.layer3(x1_)
        x1 = self.resnet1.layer4(x1)
        x2 = self.resnet2.layer3(x2_)
        x2 = self.resnet2.layer4(x2)
        x3 = self.resnet3.layer3(x3_)
        x3 = self.resnet3.layer4(x3)
        x4 = self.resnet4.layer3(x4_)
        x4 = self.resnet4.layer4(x4)

        final_size = x0.size()[2]
        x0 = F.max_pool2d(x0, kernel_size=final_size, stride=1, padding=0)
        x0 = x0.squeeze(3).squeeze(2)

        final_size = x1.size()[2]
        x1 = F.max_pool2d(x1, kernel_size=final_size, stride=1, padding=0)
        x1 = x1.squeeze(3).squeeze(2)

        final_size = x2.size()[2]
        x2 = F.max_pool2d(x2, kernel_size=final_size, stride=1, padding=0)
        x2 = x2.squeeze(3).squeeze(2)

        final_size = x3.size()[2]
        x3 = F.max_pool2d(x3, kernel_size=final_size, stride=1, padding=0)
        x3 = x3.squeeze(3).squeeze(2)

        final_size = x4.size()[2]
        x4 = F.max_pool2d(x4, kernel_size=final_size, stride=1, padding=0)
        x4 = x4.squeeze(3).squeeze(2)

        x = torch.cat([x0, x1, x2, x3, x4], dim=1)

        x = self.fc(x)
        x = F.relu(x)
        if self.keep_prob < 1.0:
            x = self.dropout(x)
        x = self.top(x)
        return x
