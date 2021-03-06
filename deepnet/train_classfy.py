
from net import SimClassfy
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import config as conf
import torch.utils.data
import torch.nn.functional as F

from dataset import val_dataloader,PathDataset
train_label_dir = conf.TRAIN_LABEL_DIR
train_datasets = PathDataset(train_label_dir,mask=0)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adjust_learning_rate_step(optimizer, learning_rate_base, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # if epoch < 2:
    #     lr = 1e-5 + (learning_rate_base-1e-6) * iteration / (epoch_size * 5)
    # else:
    #     lr = learning_rate_base * (gamma ** (step_index))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # return lr

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr *= 0.5
        param_group['lr'] = lr
    return lr

def trainBatch(net, criterion, optimizer,train_iter):
    data = train_iter.next()
    state,actions,label,_ = data

    state = state.squeeze(0).to(conf.DEVICE)
    actions = actions.squeeze(0).to(conf.DEVICE)
    label = label.squeeze(0).to(conf.DEVICE)

    preds = net(state,actions)
    sfpreds = F.softmax(preds,dim=0)
    cost = criterion(sfpreds ,label)
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def eval(model,epoch):
    model.eval()
    val_iter = iter(val_dataloader)
    count = len(val_iter)
    i = 0
    trueCount = 0
    while i < count:
        i += 1
        data = val_iter.next()
        state, actions, label, _ = data
        state = state.squeeze(0).to(conf.DEVICE)
        actions = actions.squeeze(0).to(conf.DEVICE)
        label = label.squeeze(0).to(conf.DEVICE)

        preds = model(state, actions)
        sfpreds = F.softmax(preds, dim=0)
        prediction = torch.max(sfpreds, 0)[1].item()
        answer = torch.max(label, 0)[1].item()
        if prediction == answer:
            trueCount += 1
    print('epoch = {}eval acc = {}'.format(epoch,(trueCount/count)))

def train(episodes,log_every=100, model_every=1000):
    net = SimClassfy().to(conf.DEVICE)
    criterion = nn.MSELoss().to(conf.DEVICE)
    # criterion = nn.BCELoss().to(conf.DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=conf.TRAIN_LR, betas=(0.5, 0.999),weight_decay=conf.WEIGHT_DECAY)
    # optimizer = optim.SGD(net.parameters(), lr=conf.TRAIN_LR,momentum=conf.MOMENTUM, weight_decay=conf.WEIGHT_DECAY)
    eval_iter = conf.EVAL_ITER

    # step ?????????????????????
    train_iter = iter(train_dataloader)
    epoch_size = len(train_iter)
    print('total iter = ', epoch_size)
    max_iter = episodes * epoch_size
    start_iter = 1

    epoch = 1
    stepvalues = (3 * epoch_size, 6 * epoch_size, 9 * epoch_size)
    step_index = 0
    total_loss = 0
    lr = conf.TRAIN_LR
    for iteration in range(start_iter, max_iter):
        ##???????????????
        if iteration % epoch_size == 0:
            train_iter = iter(train_dataloader)
            epoch += 1
            if epoch % 5 == 0:
                net.save('deepnet')
                print('save {} epoch'.format(epoch))
            eval(net,epoch)
            net.train()
            lr = adjust_learning_rate_step(optimizer, conf.TRAIN_LR, 0.1, epoch, step_index, iteration, epoch_size)

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate_step(optimizer, conf.TRAIN_LR, 0.1, epoch, step_index, iteration, epoch_size)
        cost = trainBatch(net, criterion, optimizer, train_iter)
        total_loss += cost
        if iteration % eval_iter == 0:
            print('ave cost={} lr = {} progress = {}'.format(total_loss / eval_iter, lr , (iteration%epoch_size) / epoch_size))
            total_loss = 0

def train_ori(epochs):
    net = DeepNet().to(conf.DEVICE)
    criterion = nn.MSELoss().to(conf.DEVICE)
    # criterion = nn.BCELoss().to(conf.DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=conf.TRAIN_LR, betas=(0.5, 0.999), weight_decay=conf.WEIGHT_DECAY)
    # optimizer = optim.SGD(net.parameters(), lr=conf.TRAIN_LR,momentum=conf.MOMENTUM, weight_decay=conf.WEIGHT_DECAY)
    eval_iter = conf.EVAL_ITER

    total_loss = 0
    for epoch in range(epochs):
        i = 0
        net.train()
        train_iter = iter(train_dataloader)
        count = len(train_iter)
        while i < count:
            i += 1
            cost = trainBatch(net, criterion, optimizer, train_iter)
            total_loss += cost
            if i % eval_iter == 0 and i > 0:
                print('ave cost={} progress = {}'.format(total_loss/eval_iter,i/count))
                total_loss = 0

        epoch += 1
        if epoch % 5 == 0:
            net.save('deepnet')
            print('save {} epoch'.format(epoch))

        eval(net,epoch)

if __name__ == '__main__':
    train(20)
