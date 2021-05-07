from net import DeepNet
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import config as conf
import torch.utils.data

from dataset import val_dataloader

def perceive():
    train_iter = iter(val_dataloader)
    i = 0
    total = len(train_iter)
    count = 0
    while i < total:
        i += 1
        net.load('deepnet')
        net.eval()

        data = train_iter.next()
        state, actions, label,_ = data

        state = state.squeeze(0).to(conf.DEVICE)
        actions = actions.squeeze(0).to(conf.DEVICE)
        label = label.squeeze(0).cpu().numpy()
        index = np.argmax(label)

        preds = net(state, actions).cpu().detach().numpy()
        pre = np.argmax(preds)

        if pre == index:
            count += 1
    print('acc={}'.format(count/total))


if __name__ == '__main__':
    net = DeepNet().to(conf.DEVICE)
    perceive()