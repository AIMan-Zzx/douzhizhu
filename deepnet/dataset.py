import numpy as np
import torch
from torch.utils.data import Dataset
import os
from deepnet.tool import *
import deepnet.config as cfg

class PathDataset(Dataset):
    def __init__(self,label_file,mask=0):
        with open(label_file, 'r') as f:
            self.nSamples = list(map(lambda line: line.strip().split(' '), f))
            self.mask = mask

    def __len__(self):
        return len(self.nSamples)

    def split_state(self,state):
        states = state.split('|')
        assert len(states) == 7
        idx = states[0]
        self_hands = states[1]
        unseen_cads = states[2]
        pre_pre_cards = states[3]
        pre_cards = states[4]
        last_cards = states[5]
        opp_cards = states[6]
        return idx,self_hands,unseen_cads,pre_pre_cards,pre_cards,last_cards,opp_cards

    def __getitem__(self, index):
        if index >= len(self):
            index = 0
        state = self.nSamples[index][0]
        label = self.nSamples[index][1]
        idx,self_hands,unseen_cads,pre_pre_cards,pre_cards,last_cards,opp_cards = \
            self.split_state(state)
        lg = str2arr(last_cards) if not str2arr(last_cards) == 'EMPTY' else []
        actions = valid_actions(str2arr(self_hands),lg)
        random.shuffle(actions)
        try:
            lable_str = str2arr(label)
            index = actions.index(lable_str)
        except:
            index = 0
        count = len(actions)
        label_onehot = label2onehot(count,index)
        tensor_actions = torch.tensor(batch_arr2onehot(actions),dtype=torch.float)
        self_hands_onehot = str2onehot(self_hands)
        unseen_cads_onehot = str2onehot(unseen_cads)
        last_cards_onehot = str2onehot(last_cards)
        opp_cards_onehot = oppHandsCount2onehot(opp_cards)#str2onehot(opp_cards,self.mask)
        state = np.concatenate((self_hands_onehot,
                               unseen_cads_onehot,
                               last_cards_onehot,
                               opp_cards_onehot,
                                opp_cards_onehot,
                                opp_cards_onehot))
        tensor_state = torch.from_numpy(state).float()
        return tensor_state,tensor_actions,label_onehot,actions


val_label_dir = cfg.VAL_LABEL_DIR
val_datasets = PathDataset(val_label_dir,mask=1.0)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=0)
