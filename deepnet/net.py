import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepnet.config as conf

class Net(nn.Module, ABC):
    def save(self,name,max_split=2):
        path = os.path.join(conf.MODEL_DIR,conf.name_dir(name,max_split))
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = '{}.pt'.format(path)
        torch.save(self.state_dict(), path)

    def load(self,name=None,abspath=None,max_split=2):
        if abspath:
            path = abspath
        else:
            path = os.path.join(conf.MODEL_DIR,conf.name_dir(name, max_split))
            path = '{}.pt'.format(path)
        map_location = 'cpu' if conf.DEVICE.type == 'cpu' else None
        static_dict = torch.load(path, map_location)
        self.load_state_dict(static_dict)
        self.eval()
        print("Loaded model from {}.".format(path))
'''
class DeepNet(Net):
    def __init__(self,out_chanel = 128):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(7,out_chanel,(1,1),(1,4))
        self.conv2 = nn.Conv2d(7,out_chanel,(1,2),(1,4))
        self.conv3 = nn.Conv2d(7,out_chanel,(1,3),(1,4))
        self.conv4 = nn.Conv2d(7,out_chanel,(1,4),(1,4))
        self.convs = (self.conv1,self.conv2,self.conv3,self.conv4)
        self.conv_s = nn.Conv2d(7,out_chanel,(15,1),1)
        self.pool = nn.MaxPool2d((1,4))
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(out_chanel*(15+4),out_chanel)
        self.fc2 = nn.Linear(out_chanel,1)

    def forward(self, state,actions):
        if state.dim() == 3:
            state = state.unsqueeze(0).repeat((actions.shape[0],1,1,1))
        actions = actions.unsqueeze(1)
        state_action = torch.cat((state,actions),dim=1)
        res = [f(state_action) for f in self.convs]
        x = torch.cat(res, -1)
        x = self.pool(x)
        x = F.relu(x)
        x = x.view(actions.shape[0],-1)

        x_s = self.conv_s(state_action).view(actions.shape[0],-1)
        x_s = F.relu(x_s)
        x = torch.cat([x,x_s],-1)
        # x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
class DeepNet(Net):
    def __init__(self,out_chanel = 64):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(7,out_chanel,(1,1),(1,4))
        self.conv2 = nn.Conv2d(7,out_chanel,(1,2),(1,4))
        self.conv3 = nn.Conv2d(7,out_chanel,(1,3),(1,4))
        self.conv4 = nn.Conv2d(7,out_chanel,(1,4),(1,4))
        self.convs = (self.conv1,self.conv2,self.conv3,self.conv4)
        self.conv_s = nn.Conv2d(7,out_chanel,(15,1),1)
        self.pool = nn.MaxPool2d((1,4))
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(out_chanel)
        self.fc1 = nn.Linear(out_chanel*(15+1),out_chanel)
        self.fc2 = nn.Linear(out_chanel,1)

    def forward(self, state,actions):
        if state.dim() == 3:
            state = state.unsqueeze(0).repeat((actions.shape[0],1,1,1))
        actions = actions.unsqueeze(1)
        state_action = torch.cat((state,actions),dim=1)
        res = [f(state_action) for f in self.convs]
        x = torch.cat(res, -1)
        x = self.bn(x)
        x = self.pool(x)
        # x = F.relu(x)#x = nn.PReLU(x)
        x = x.view(actions.shape[0],-1)

        x_s = self.conv_s(state_action)
        x_s = self.bn(x_s)
        x_s = self.pool(x_s)
        # x_s = F.relu(x_s)#x_s = nn.PReLU(x_s)
        x_s = x_s.view(actions.shape[0], -1)
        x = torch.cat([x,x_s],-1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))#x = nn.PReLU(self.fc1(x))
        x = self.fc2(x)
        return x

class SimClassfy(Net):
    def __init__(self,out_chanel = 64):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(7,out_chanel,(1,1),(1,4))
        self.conv2 = nn.Conv2d(7,out_chanel,(1,2),(1,4))
        self.conv3 = nn.Conv2d(7,out_chanel,(1,3),(1,4))
        self.conv4 = nn.Conv2d(7,out_chanel,(1,4),(1,4))
        self.convs = (self.conv1,self.conv2,self.conv3,self.conv4)
        self.conv_s = nn.Conv2d(7,out_chanel,(15,1),1)
        self.pool = nn.MaxPool2d((1,4))
        # self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(out_chanel)
        self.fc1 = nn.Linear(out_chanel*(15+1),out_chanel)# self.fc1 = nn.Linear(out_chanel*(15+1),out_chanel)
        self.fc2 = nn.Linear(out_chanel,1)

    def process(self,chanel):
        chanel = chanel.unsqueeze(0)
        res = [f(chanel) for f in self.convs]
        x = torch.cat(res, -1)
        x = self.bn(x)
        x = self.pool(x)
        x = F.relu(x)
        x = x.view(1, -1)

        x_s = self.conv_s(chanel)
        x_s = self.bn(x_s)
        x_s = self.pool(x_s)
        x_s = F.relu(x_s)
        x_s = x_s.view(1, -1)
        x = torch.cat([x, x_s], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, state,actions):
        if state.dim() == 3:
            state = state.unsqueeze(0).repeat((actions.shape[0],1,1,1))
        actions = actions.unsqueeze(1)
        state_action = torch.cat((state,actions),dim=1)
        size = state_action.shape[0]
        res = [self.process(state_action[i]) for i in range(size)]
        x = torch.cat(res, 0)
        return x

class ThreeDeepNet(Net):
    def __init__(self,out_chanel = 256):
        super(Net,self).__init__()
        self.conv1 = nn.Conv3d(7,out_chanel,(1,1,1),(1,1,4))
        self.conv2 = nn.Conv3d(7,out_chanel,(1,1,2),(1,1,4))
        self.conv3 = nn.Conv3d(7,out_chanel,(1,1,3),(1,1,4))
        self.conv4 = nn.Conv3d(7,out_chanel,(1,1,4),(1,1,4))
        self.convs = (self.conv1,self.conv2,self.conv3,self.conv4)
        self.conv_s = nn.Conv3d(7,out_chanel,(1,15,1),1)
        self.pool = nn.MaxPool3d((1,1,4))
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm3d(out_chanel)
        self.fc1 = nn.Linear(out_chanel*(15+1),out_chanel)
        self.fc2 = nn.Linear(out_chanel,1)

        self.Att = Attention_block(F_g=1, F_l=6, F_int=out_chanel)
        self.Att1 = Attention_block(F_g=6, F_l=6, F_int=out_chanel)

    def forward(self, state,actions):
        if state.dim() == 3:
            state = state.unsqueeze(0).repeat((actions.shape[0],1,1,1))
        actions = actions.unsqueeze(1)
        state1 = self.Att1(state, state)
        actions = self.Att(state, actions)
        state_action = torch.cat((state1,actions),dim=1)
        state_action = state_action.transpose(0,1)
        state_action = state_action.unsqueeze(0)
        res = [f(state_action) for f in self.convs]
        x = torch.cat(res, -1)
        x = self.bn(x)
        x = self.pool(x)
        x = F.relu(x)
        x = x.squeeze(0).transpose(0,1).contiguous().squeeze(-1)
        x = x.view(actions.shape[0],-1)

        x_s = self.conv_s(state_action)
        x_s = self.bn(x_s)
        x_s = self.pool(x_s)
        x_s = F.relu(x_s)
        x_s = x_s.squeeze(0).transpose(0, 1).contiguous().squeeze(-1)
        x_s = x_s.view(actions.shape[0], -1)
        x = torch.cat([x,x_s],-1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttNet(Net):
    def __init__(self,out_chanel = 256):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(7,out_chanel,(1,1),(1,4))
        self.conv2 = nn.Conv2d(7,out_chanel,(1,2),(1,4))
        # self.conv3 = nn.Conv2d(7,out_chanel,(1,3),(1,4))
        # self.conv4 = nn.Conv2d(7,out_chanel,(1,4),(1,4))
        # self.convs = (self.conv1,self.conv2,self.conv3,self.conv4)
        self.convs = (self.conv1,self.conv2)
        self.conv_s = nn.Conv2d(7,out_chanel,(15,1),1)
        self.pool = nn.MaxPool2d((1,2))# self.pool = nn.MaxPool2d((1,4))
        self.pool1 = nn.MaxPool2d((1, 4))

        self.drop = nn.Dropout(0.5)
        # self.bn = nn.BatchNorm2d(out_chanel)
        # self.fc1 = nn.Linear(out_chanel*(15+1),1)
        self.fc1 = nn.Linear(out_chanel*(15+1),out_chanel)
        self.fc2 = nn.Linear(out_chanel,1)

        # self.Att = Attention_block(F_g=1, F_l=4, F_int=out_chanel)
        # self.Att1 = Attention_block(F_g=4, F_l=4, F_int=out_chanel)

        # self.Att2 = Attention_block(F_g=out_chanel, F_l=1, F_int=out_chanel)
        # self.Att3 = Attention_block(F_g=out_chanel, F_l=6, F_int=out_chanel)

        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()

    def forward(self, state,actions):
        if state.dim() == 3:
            state = state.unsqueeze(0).repeat((actions.shape[0],1,1,1))
        actions = actions.unsqueeze(1)

        # state = self.Att1(state, state)
        # actions = self.Att(state,actions)

        state_action = torch.cat((state,actions),dim=1)
        res = [f(state_action) for f in self.convs]
        x = torch.cat(res, -1)


        # x = self.bn(x)
        x = self.pool(x)
        x = F.relu(x)
        x = x.view(actions.shape[0],-1)

        x_s = self.conv_s(state_action)
        # x_s = self.bn(x_s)
        x_s = self.pool1(x_s)
        x_s = F.relu(x_s)
        x_s = x_s.view(actions.shape[0], -1)
        x = torch.cat([x,x_s],-1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
