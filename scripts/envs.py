import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from datetime import datetime
import numpy as np
from card import Card, Category, CardGroup, action_space
from util import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize

from tensorpack import *
from env import Env as CEnv
from mct import mcsearch, mcsearch_maybecards, CCard, CCardGroup, CCategory, get_action, \
    getActionspaceCount,mcsearch_god,get_topk_actions,getActionspace
from TensorPack.MA_Hierarchical_Q.env import Env
from TensorPack.MA_Hierarchical_Q.predictor import Predictor
from TensorPack.MA_Hierarchical_Q.DQNModel import Model

from deepnet.dataset import PathDataset
import torch
from deepnet.tool import *
import deepnet.config as cfg
from deepnet.net import DeepNet,AttNet,ThreeDeepNet
import deepnet.config as conf

# from tools import make_sqit_cards


limit_card_num = 2
weight_path = os.path.join(ROOT_PATH, 'pretrained_model/model-302500')

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

class MCTEnv(Env):

    def __init__(self):
        super(MCTEnv,self).__init__()
        self.predicter = Predict('attnet0')

    def step(self, intention):
        player, done = super().step(intention)
        if player != self.agent_names[0]:
            return 1, done
        else:
            return -1, done

    def manual_step_auto(self):
        def char2ccardgroup(chars):
            cg = CardGroup.to_cardgroup(chars)
            ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
            return ccg

        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]

        handcards_char = self.get_curr_handcards()
        print('农民手牌：', handcards_char)
        manual_input = input("农民出牌:")
        intention_maybecards = list(manual_input)
        return self.step(intention_maybecards)

    def write2file(self,filename,cur,selfhandcards,unseen_cards,pre_pre_hands,pre_hands,last_hands,opphandcards,action):
        # unseen_cards = sorted(unseen_cards[:], key=lambda k: Card.cards_to_value[k])
        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]

        with open(filename, 'a') as a:
            cselfhandcards = "".join(selfhandcards)
            cunseen_cards = "".join(unseen_cards)
            cpre_pre_hands = "".join(ccardgroup2char(pre_pre_hands)) if len(pre_pre_hands.cards) > 0 else 'EMPTY'
            cpre_hands = "".join(ccardgroup2char(pre_hands)) if len(pre_hands.cards) > 0 else 'EMPTY'
            clast_hands = "".join(ccardgroup2char(last_hands)) if len(last_hands.cards) > 0 else 'EMPTY'
            copphandcards = "".join(opphandcards)
            caction = "".join(action) if len(action) > 0 else 'EMPTY'

            content = str(cur) + '|' + cselfhandcards + '|' + cunseen_cards + '|' \
                      + cpre_pre_hands + '|' + cpre_hands + '|' \
                      + clast_hands + '|' + copphandcards + ' ' \
                      + caction
            a.write(content + '\n')

    def step_auto(self):
        def char2ccardgroup(chars):
            cg = CardGroup.to_cardgroup(chars)
            ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
            return ccg

        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]



        handcards_char = self.get_curr_handcards()
        chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
        player_idx = self.get_current_idx()
        # unseen_cards = self.player_cards[self.agent_names[(player_idx + 1) % 3]] + self.player_cards[self.agent_names[(player_idx + 2) % 3]]


        next_handcards_cnt = len(self.player_cards[self.agent_names[(player_idx + 1) % 2]])

        last_cg = char2ccardgroup(self.get_last_outcards())

        action_count = getActionspaceCount(chandcards,last_cg,
                             (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2,
                             (self.agent_names.index(self.controller) - self.agent_names.index(self.lord) + 2) % 2)

        cur = (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2
        cpre_handcards = []
        cpre_pre_handcards = []
        if len(self.mcts_histories[self.agent_names[cur]]) > 0:
            cpre_handcards = self.mcts_histories[self.agent_names[cur]][-1]
        if len(self.mcts_histories[self.agent_names[(cur + 1) % 2]]) > 1:
            cpre_pre_handcards = self.mcts_histories[self.agent_names[(cur + 1) % 2]][-2]
        pre_handcards = char2ccardgroup(cpre_handcards)
        pre_pre_handcards = char2ccardgroup(cpre_pre_handcards)
        unseen_cards = self.player_cards[self.agent_names[(player_idx + 1) % 2]] + self.extra_cards
        cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]

        opphands = self.player_cards[self.agent_names[(player_idx + 1) % 2]]
        copphands = [CCard(to_value(c) - 3) for c in opphands]

        path = cfg.PREDICT_ENV_LABEL_DIR
        if os.path.exists(path):
            os.remove(path)
        self.write2file(path, cur, handcards_char, unseen_cards, pre_pre_handcards, pre_handcards, last_cg, opphands,
                        [])
        predict_label_dir = cfg.PREDICT_ENV_LABEL_DIR
        predict_datasets = PathDataset(predict_label_dir)
        predict_dataloader = torch.utils.data.DataLoader(predict_datasets, batch_size=1, shuffle=True, num_workers=0)
        action = self.predicter.perceive(predict_dataloader)
        actions = [a[0] for a in action]
        caction_maybecards = [] if action[0] == 'EMPTY' else list(actions)
        return self.step(caction_maybecards)

        next_handcards_cnt = len(opphands)
        caction_maybecards = mcsearch_maybecards(chandcards,[], cunseen_cards,next_handcards_cnt, last_cg,
                       (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2,
                       (self.agent_names.index(self.controller) - self.agent_names.index(self.lord) + 2) % 2, 15, 300, 3000, 2, limit_card_num)
        intention_maybecards = ccardgroup2char(caction_maybecards)



    '''
    def step_auto(self):
        def char2ccardgroup(chars):
            cg = CardGroup.to_cardgroup(chars)
            ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
            return ccg

        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]



        handcards_char = self.get_curr_handcards()
        chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
        player_idx = self.get_current_idx()
        # unseen_cards = self.player_cards[self.agent_names[(player_idx + 1) % 3]] + self.player_cards[self.agent_names[(player_idx + 2) % 3]]


        next_handcards_cnt = len(self.player_cards[self.agent_names[(player_idx + 1) % 2]])

        last_cg = char2ccardgroup(self.get_last_outcards())

        action_count = getActionspaceCount(chandcards,last_cg,
                             (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2,
                             (self.agent_names.index(self.controller) - self.agent_names.index(self.lord) + 2) % 2)

        cur = (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2
        cpre_handcards = []
        cpre_pre_handcards = []
        if len(self.mcts_histories[self.agent_names[cur]]) > 0:
            cpre_handcards = self.mcts_histories[self.agent_names[cur]][-1]
        if len(self.mcts_histories[self.agent_names[(cur + 1) % 2]]) > 1:
            cpre_pre_handcards = self.mcts_histories[self.agent_names[(cur + 1) % 2]][-2]
        pre_handcards = char2ccardgroup(cpre_handcards)
        pre_pre_handcards = char2ccardgroup(cpre_pre_handcards)
        unseen_cards = self.player_cards[self.agent_names[(player_idx + 1) % 2]] + self.extra_cards
        cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]

        opphands = self.player_cards[self.agent_names[(player_idx + 1) % 2]]
        copphands = [CCard(to_value(c) - 3) for c in opphands]

        intention_maybecards = []
        valid_actions = []
        # caction_maybecards = mcsearch_god(chandcards, copphands, last_cg,
        #                (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2,
        #                (self.agent_names.index(self.controller) - self.agent_names.index(self.lord) + 2) % 2, 15, 1, 2000, 2, limit_card_num)
        next_handcards_cnt = len(opphands)
        caction_maybecards = mcsearch_maybecards(chandcards,[], cunseen_cards,next_handcards_cnt, last_cg,
                       (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2,
                       (self.agent_names.index(self.controller) - self.agent_names.index(self.lord) + 2) % 2, 15, 300, 3000, 2, limit_card_num)
        intention_maybecards = ccardgroup2char(caction_maybecards)
            #
            # valid_actions = getActionspace(chandcards,last_cg,
            #                                (self.agent_names.index(self.curr_player) - self.agent_names.index(self.lord) + 2) % 2,
            #                                (self.agent_names.index(self.controller) - self.agent_names.index(self.lord) + 2) % 2)

        if action_count > 1:
            self.write2file('1.txt', cur, handcards_char, unseen_cards, pre_pre_handcards, pre_handcards, last_cg,
                            opphands, intention_maybecards)
        # if len(intention_maybecards) == 0:
        #     if action_count > 1:
        #         self.write2file('1.txt', cur, handcards_char, unseen_cards, pre_pre_handcards, pre_handcards, last_cg,
        #                         opphands, intention_maybecards)
        # else:
        #     self.write2file('1.txt', cur, handcards_char, unseen_cards, pre_pre_handcards, pre_handcards, last_cg,
        #                     opphands, intention_maybecards)
        if len(intention_maybecards) == 0 and len(last_cg.cards) == 0:
            print('error')
            pass
        return self.step(intention_maybecards)
    '''

class RandomEnv(Env):
    def step(self, intention):
        # print(self.get_curr_handcards())
        # print(intention)
        player, done = super().step(intention)
        if player != self.agent_names[0]:
            return 1, done
        else:
            return -1, done

    def step_auto(self):
        mask = get_mask(self.get_curr_handcards(), action_space, self.get_last_outcards())
        intention = np.random.choice(action_space, 1, p=mask / mask.sum())[0]
        return self.step(intention)


class CDQNEnv(Env):
    def __init__(self, weight_path):
        super().__init__()
        agent_names = ['agent%d' % i for i in range(1, 4)]
        # model = Model(agent_names, (1000, 21, 256 + 256 * 2 + 120), 'Double', (1000, 21), 0.99)
        model = Model(agent_names, (1000, 21, 256 + 256  + 60), 'Double', (1000, 21), 0.99)
        self.predictors = {n: Predictor(OfflinePredictor(PredictConfig(
            model=model,
            session_init=SaverRestore(weight_path),
            input_names=[n + '/state', n + '_comb_mask', n + '/fine_mask'],
            output_names=[n + '/Qvalue'])), num_actions=(1000, 21)) for n in self.get_all_agent_names()}

    def step(self, intention):
        # print(intention)
        player, done = super().step(intention)
        if player != self.agent_names[0]:
            return 1, done
        else:
            return -1, done

    def step_auto(self):
        handcards = self.get_curr_handcards()
        last_two_cards = self.get_last_two_cards()
        prob_state = self.get_state_prob()
        intention = self.predictors[self.get_curr_agent_name()].predict(handcards, last_two_cards, prob_state)
        return self.step(intention)

class RHCPEnv(CEnv):
    def __init__(self):
        super().__init__()
        self.agent_names = ['agent1', 'agent2','agent3']

    def prepare(self):
        super().prepare()
        self.lord = self.agent_names[self.get_current_idx()]
        self.controller = self.lord
        # print('lord is ', self.lord, self.get_role_ID())

    @property
    def curr_player(self):
        return self.agent_names[self.get_current_idx()]

    @property
    def player_cards(self):
        other_two = self.get_last_two_handcards()
        curr_idx = self.get_current_idx()
        return {
            self.agent_names[(curr_idx + 2) % 3]: to_char(other_two[1]),
            self.agent_names[(curr_idx + 1) % 3]: to_char(other_two[0]),
            self.agent_names[curr_idx]: self.get_curr_handcards()
        }

    def get_current_idx(self):
        return super().get_curr_ID()

    def get_last_outcards(self):
        return to_char(super().get_last_outcards())

    def get_last_two_cards(self):
        last_two_cards = super().get_last_two_cards()
        last_two_cards = [to_char(c) for c in last_two_cards]
        return last_two_cards

    def get_curr_handcards(self):
        return to_char(super().get_curr_handcards())

    def step(self, intention):
        # print(intention)
        idx = self.get_current_idx()
        print('地主：', self.player_cards[self.agent_names[idx]])
        print('地主:', 'gives', intention, self.controller)
        print('\n')

        r, done = self.step2(to_value(intention))

        return r, done

    def step_auto(self):
        idx = self.get_current_idx()
        # print(idx)
        id = '上家：'
        if idx ==1:
            id= '下家：'
        print(id,self.player_cards[self.agent_names[idx]])
        intention, r = super().step2_auto()
        intention = to_char(intention)
        print(id, 'gives', intention, self.controller)
        print('\n')

        if len(intention) > 0:
            self.controller = self.agent_names[idx]
        assert np.all(self.get_state_prob() >= 0) and np.all(self.get_state_prob() <= 1)
        # print(intention)
        return r, r != 0

def make_env(which):
    if which == 'RHCP':
        return RHCPEnv()
    elif which == 'RANDOM':
        return RandomEnv()
    elif which == 'CDQN':
        return CDQNEnv(weight_path)
    elif which == 'MCT':
        return MCTEnv()
    else:
        raise Exception('env type not supported')
