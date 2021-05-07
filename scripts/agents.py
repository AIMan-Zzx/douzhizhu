import os
import sys
from deepnet.dataset import PathDataset
import torch

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from datetime import datetime
import numpy as np
from card import Card, Category, CardGroup, action_space
from util import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize

from datetime import datetime
from tensorpack import *
from env import Env as CEnv
from mct import mcsearch, CCard, CCardGroup, CCategory, mcsearch_god, getActionspaceCount,get_topk_actions,getActionspace,mcsearch_maybecards
from TensorPack.MA_Hierarchical_Q.predictor import Predictor
from TensorPack.MA_Hierarchical_Q.DQNModel import Model
# from tools import make_sqit_cards

from deepnet.tool import *
import deepnet.config as cfg
from deepnet.net import DeepNet,AttNet,ThreeDeepNet
import deepnet.config as conf

limit_card_num = 2

class Predict:
    def __init__(self,path):
        self.net = AttNet().to(conf.DEVICE)
        # self.net = ThreeDeepNet().to(conf.DEVICE)
        self.net.load(path)
        self.net.eval()

    def perceive(self,predict_dataloader):
        train_iter = iter(predict_dataloader)
        data = train_iter.next()
        state, actions, _,ori_actions = data

        state = state.squeeze(0).to(conf.DEVICE)
        actions = actions.squeeze(0).to(conf.DEVICE)
        preds = self.net(state, actions).cpu().detach()
        # values, indices = preds.view(-1).topk(3,dim=0)
        preds = preds.numpy()
        pre = np.argmax(preds)
        return ori_actions[pre]

class Agent:
    def __init__(self, role_id):
        self.role_id = role_id

    def intention(self, env):
        pass

    def write2file(self,filename,cur,selfhandcards,unseen_cards,pre_pre_hands,pre_hands,last_hands,opphandcards,action):
        # unseen_cards = sorted(unseen_cards[:], key=lambda k: Card.cards_to_value[k])
        with open(filename,'a') as a:
            cselfhandcards = "".join(selfhandcards)
            cunseen_cards = "".join(unseen_cards)
            cpre_pre_hands = "".join(ccardgroup2char(pre_pre_hands)) if len(pre_pre_hands.cards) > 0 else 'EMPTY'
            cpre_hands = "".join(ccardgroup2char(pre_hands)) if len(pre_hands.cards) > 0 else 'EMPTY'
            clast_hands = "".join(ccardgroup2char(last_hands)) if len(last_hands.cards) > 0 else 'EMPTY'
            copphandcards = "".join(opphandcards) if len(opphandcards) > 0 else 'EMPTY'
            caction = "".join(action) if len(action) > 0 else 'EMPTY'

            content = str(cur) + '|' + cselfhandcards + '|' + cunseen_cards + '|'\
                      + cpre_pre_hands + '|' + cpre_hands + '|'\
                      + clast_hands + '|' + copphandcards + ' '\
                      + caction
            a.write(content + '\n')


class RandomAgent(Agent):
    def intention(self, env):
        mask = get_mask(env.get_curr_handcards(), action_space, env.get_last_outcards())
        intention = np.random.choice(action_space, 1, p=mask / mask.sum())[0]
        return intention


class RHCPAgent(Agent):
    def intention(self, env):
        intention = to_char(CEnv.step_auto_static(Card.char2color(env.get_curr_handcards()), to_value(env.get_last_outcards())))
        # print('rhcp handcards:',env.get_curr_handcards())
        # print('rhcp give cards：', intention)
        return intention

    def search(self,handcards,last_cards):
        intention = to_char(
            CEnv.step_auto_static(Card.char2color(handcards), to_value(last_cards)))
        # print('rhcp handcards:', handcards)
        # print('rhcp give cards：', intention)
        return intention

class DEEPNETAgent(Agent):
    def __init__(self, role_id):
        super().__init__(role_id)
        self.predict = Predict('attnet1')

    def intention(self, env):
        def char2ccardgroup(chars):
            cg = CardGroup.to_cardgroup(chars)
            ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
            return ccg

        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]

        handcards_char = env.get_curr_handcards()
        chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
        player_idx = env.get_current_idx()

        last_cg = char2ccardgroup(env.get_last_outcards())

        action_count = getActionspaceCount(chandcards, last_cg,
                                           (env.agent_names.index(env.curr_player) - env.agent_names.index(
                                               env.lord) + 2) % 2,
                                           (env.agent_names.index(env.controller) - env.agent_names.index(
                                               env.lord) + 2) % 2)

        cur = (env.agent_names.index(env.curr_player) - env.agent_names.index(env.lord) + 2) % 2
        cpre_handcards = []
        cpre_pre_handcards = []
        if len(env.mcts_histories[env.agent_names[cur]]) > 0:
            cpre_handcards = env.mcts_histories[env.agent_names[cur]][-1]
        if len(env.mcts_histories[env.agent_names[(cur + 1) % 2]]) > 1:
            cpre_pre_handcards = env.mcts_histories[env.agent_names[(cur + 1) % 2]][-2]
        pre_handcards = char2ccardgroup(cpre_handcards)
        pre_pre_handcards = char2ccardgroup(cpre_pre_handcards)

        unseen_cards = env.player_cards[env.agent_names[(player_idx + 1) % 2]] + env.extra_cards
        cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]

        opphands = env.player_cards[env.agent_names[(player_idx + 1) % 2]]

        path = cfg.PREDICT_AGENT_LABEL_DIR
        if os.path.exists(path):
            os.remove(path)
        self.write2file(path, cur, handcards_char, unseen_cards, pre_pre_handcards, pre_handcards, last_cg,opphands, [])
        predict_label_dir = cfg.PREDICT_AGENT_LABEL_DIR
        predict_datasets = PathDataset(predict_label_dir)
        predict_dataloader = torch.utils.data.DataLoader(predict_datasets, batch_size=1, shuffle=True, num_workers=0)
        action = self.predict.perceive(predict_dataloader)
        actions = [a[0] for a in action]
        caction_maybecards = [] if action[0] == 'EMPTY' else list(actions)
        return caction_maybecards


class MCTAgent(Agent):
    def intention(self, env):
        def char2ccardgroup(chars):
            cg = CardGroup.to_cardgroup(chars)
            ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
            return ccg

        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]


        handcards_char = env.get_curr_handcards()
        chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
        player_idx = env.get_current_idx()

        last_cg = char2ccardgroup(env.get_last_outcards())

        action_count = getActionspaceCount(chandcards, last_cg,
                                           (env.agent_names.index(env.curr_player) - env.agent_names.index(
                                               env.lord) + 2) % 2,
                                           (env.agent_names.index(env.controller) - env.agent_names.index(
                                               env.lord) + 2) % 2)

        cur = (env.agent_names.index(env.curr_player) - env.agent_names.index(env.lord) + 2) % 2
        cpre_handcards = []
        cpre_pre_handcards = []
        if len(env.mcts_histories[env.agent_names[cur]]) > 0:
            cpre_handcards = env.mcts_histories[env.agent_names[cur]][-1]
        if len(env.mcts_histories[env.agent_names[(cur + 1) % 2]]) > 1:
            cpre_pre_handcards = env.mcts_histories[env.agent_names[(cur + 1) % 2]][-2]
        pre_handcards = char2ccardgroup(cpre_handcards)
        pre_pre_handcards = char2ccardgroup(cpre_pre_handcards)

        unseen_cards = env.player_cards[env.agent_names[(player_idx + 1) % 2]] + env.extra_cards
        cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]

        opphands = env.player_cards[env.agent_names[(player_idx + 1) % 2]]
        copphands = [CCard(to_value(c) - 3) for c in opphands]

        intention_maybecards = []
        valid_actions = []

        # caction_maybecards = mcsearch_god(chandcards, copphands, last_cg,
        #                                          (env.agent_names.index(env.curr_player) - env.agent_names.index(
        #                                              env.lord) + 2) % 2,
        #                                          (env.agent_names.index(env.controller) - env.agent_names.index(
        #                                              env.lord) + 2) % 2, 15, 1, 2000, 2, limit_card_num)
        next_handcards_cnt = len(opphands)
        caction_maybecards = mcsearch_maybecards(chandcards, [], cunseen_cards, next_handcards_cnt, last_cg,
                                                 (env.agent_names.index(env.curr_player) - env.agent_names.index(
                                                     env.lord) + 2) % 2,
                                                 (env.agent_names.index(env.controller) - env.agent_names.index(
                                                     env.lord) + 2) % 2, 15, 300, 3000, 2, limit_card_num)

        intention_maybecards = ccardgroup2char(caction_maybecards)
            # valid_actions = getActionspace(chandcards, last_cg,
            #                                (env.agent_names.index(env.curr_player) - env.agent_names.index(env.lord) + 2) % 2,
            #                                (env.agent_names.index(env.controller) - env.agent_names.index(env.lord) + 2) % 2)
        # if len(intention_maybecards) == 0:
        #     if action_count > 1:
        #         self.write2file('0.txt',cur,handcards_char,unseen_cards,pre_pre_handcards,pre_handcards,last_cg,opphands,intention_maybecards)
        # else:
        #     self.write2file('0.txt',cur,handcards_char,unseen_cards,pre_pre_handcards,pre_handcards,last_cg,opphands,intention_maybecards)
        if action_count > 1:
            self.write2file('0.txt', cur, handcards_char, unseen_cards, pre_pre_handcards, pre_handcards, last_cg,
                            opphands, intention_maybecards)
        if len(intention_maybecards) == 0 and len(last_cg.cards) == 0:
            print('error')
            pass

        return intention_maybecards


def make_agent(which, role_id):
    if which == 'DEEPNET':
        return DEEPNETAgent(role_id)
    elif which == 'RHCP':
        return RHCPAgent(role_id)
    elif which == 'RANDOM':
        return RandomAgent(role_id)
    elif which == 'MCT':
        return MCTAgent(role_id)
    else:
        raise Exception('env type not supported')


def ccardgroup2char(cg):
    return [to_char(int(c) + 3) for c in cg.cards]

def char2ccardgroup(chars):
    cg = CardGroup.to_cardgroup(chars)
    ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
    return ccg

def ztes():
    # vector < int > hands({5, 7, 8, 9, 12});
    # handcards_char [= ['$','2','K','Q','J','10','9','8','8','8','5','5']
    handcards_char = ['8','9','10','J','Q','2']

    chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
    # vector < int > unseen_hands({2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 10, 12});

    # unseen_cards = ['A','K','9','J','Q','K','6','7','9','J','Q','K','A','5','6','7','8','9','10','J','Q','*']
    unseen_cards = ['5','6','7','7','8','8','9','9','10','J','Q','K','K','2']

    unseen_cards = sorted(unseen_cards)
    cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]
    last_cg = char2ccardgroup([])

    pre_cg = char2ccardgroup(['5'])

    caction_maybecards = mcsearch(chandcards, cunseen_cards, 5, pre_cg,last_cg,0,1, 10, 100, 5000, 2, limit_card_num)
    intention_maybecards = ccardgroup2char(caction_maybecards)
    # agent = make_agent('RHCP',1)
    # intention_maybecards = agent.search(handcards_char,[])
    print(intention_maybecards)

def test2():
    handcards_char = ['6','6','A','A']
    chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
    unseen_cards = ['7','7','K','K','K','K']
    cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]
    last_cg = char2ccardgroup(['5',"5"])
    caction_maybecards = mcsearch_maybecards(chandcards, [], cunseen_cards, 4, last_cg,1,0, 10, 100, 5000, 1, 2)
    intention_maybecards = ccardgroup2char(caction_maybecards)
    print(intention_maybecards)

if __name__ == '__main__':

    test()
