from deepnet.net import DeepNet
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import deepnet.config as conf
import torch.utils.data

class Predict:
    def __init__(self,path):
        self.net = DeepNet().to(conf.DEVICE)
        self.net.load(path)
        self.net.eval()

    def perceive(self,predict_dataloader):
        train_iter = iter(predict_dataloader)
        data = train_iter.next()
        state, actions, _,actions = data

        state = state.squeeze(0).to(conf.DEVICE)
        actions = actions.squeeze(0).to(conf.DEVICE)
        preds = self.net(state, actions).cpu().detach().numpy()
        pre = np.argmax(preds)
        return actions[pre]

