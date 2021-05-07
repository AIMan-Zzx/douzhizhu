from collections import Counter
from enum import Enum
import numpy as np
import itertools
import functools
import numpy as np
import deepnet.config as cfg
import random
# import os
# import sys
# FILE_PATH = os.path.dirname(os.path.abspath(__file__))
# ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
# sys.path.append(ROOT_PATH)
# # sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
# sys.path.insert(0, ROOT_PATH)


from mct import mcsearch, mcsearch_maybecards, CCard, CCardGroup, CCategory, get_action, \
    getActionspaceCount,mcsearch_god,get_topk_actions,getActionspace
class Category:
    EMPTY = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRIC = 4
    THREE_ONE = 5
    THREE_TWO = 6
    SINGLE_LINE = 7
    DOUBLE_LINE = 8
    TRIPLE_LINE = 9
    THREE_ONE_LINE = 10
    THREE_TWO_LINE = 11
    BIGBANG = 12
    FOUR_TAKE_ONE = 13
    FOUR_TAKE_TWO = 14


class Card:
    cards = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '*', '$']
    np_cards = np.array(cards)
    # full_cards = [x for pair in zip(cards, cards, cards, cards) for x in pair if x not in ['*', '$']]
    # full_cards += ['*', '$']
    cards_to_onehot_idx = dict((x, i * 4) for (i, x) in enumerate(cards))
    cards_to_onehot_idx['*'] = 52
    cards_to_onehot_idx['$'] = 53
    cards_to_value = dict(zip(cards, range(len(cards))))
    value_to_cards = dict((v, c) for (c, v) in cards_to_value.items())

    def __init__(self):
        pass

    @staticmethod
    def char2onehot(cards):
        counts = Counter(cards)
        onehot = np.zeros(54)
        for x in cards:
            if x in ['*', '$']:
                onehot[Card.cards_to_onehot_idx[x]] = 1
            else:
                subvec = np.zeros(4)
                subvec[:counts[x]] = 1
                onehot[Card.cards_to_onehot_idx[x]:Card.cards_to_onehot_idx[x] + 4] = subvec
        return onehot

    @staticmethod
    def char2onehot60(cards):
        counts = Counter(cards)
        onehot = np.zeros(60, dtype=np.int32)
        for x in cards:
            subvec = np.zeros(4)
            subvec[:counts[x]] = 1
            onehot[Card.cards.index(x) * 4:Card.cards.index(x) * 4 + 4] = subvec
        return onehot

    @staticmethod
    def val2onehot(cards):
        chars = [Card.cards[i - 3] for i in cards]
        return Card.char2onehot(chars)

    @staticmethod
    def val2onehot60(cards):
        counts = Counter(cards)
        onehot = np.zeros(60)
        for x in cards:
            idx = (x - 3) * 4
            subvec = np.zeros(4)
            subvec[:counts[x]] = 1
            onehot[idx:idx + 4] = subvec
        return onehot

    # convert char to 0-56 color cards
    @staticmethod
    def char2color(cards):
        result = np.zeros([len(cards)])
        mask = np.zeros([57])
        for i in range(len(cards)):
            ind = Card.cards.index(cards[i]) * 4
            while mask[ind] == 1:
                ind += 1
            mask[ind] = 1
            result[i] = ind

        return result

    @staticmethod
    def onehot2color(cards):
        result = []
        for i in range(len(cards)):
            if cards[i] == 0:
                continue
            if i == 53:
                result.append(56)
            else:
                result.append(i)
        return np.array(result)

    @staticmethod
    def onehot2char(cards):
        result = []
        for i in range(len(cards)):
            if cards[i] == 0:
                continue
            if i == 53:
                result.append(Card.cards[14])
            else:
                result.append(Card.cards[i // 4])
        return result

    @staticmethod
    def onehot2val(cards):
        result = []
        for i in range(len(cards)):
            if cards[i] == 0:
                continue
            if i == 53:
                result.append(17)
            else:
                result.append(i // 4 + 3)
        return result

    @staticmethod
    def char2value_3_17(cards):
        result = []
        if type(cards) is list or type(cards) is range:
            for c in cards:
                result.append(Card.cards_to_value[c] + 3)
            return np.array(result)
        else:
            return Card.cards_to_value[cards] + 3

    @staticmethod
    def to_value(card):
        if type(card) is list or type(card) is range:
            val = 0
            for c in card:
                val += Card.cards_to_value[c]
            return val
        else:
            return Card.cards_to_value[card]

    @staticmethod
    def to_cards(values):
        if type(values) is list or type(values) is range:
            cards = []
            for v in values:
                cards.append(Card.value_to_cards[v])
            return cards
        else:
            return Card.value_to_cards[values]

    @staticmethod
    def to_cards_from_3_17(values):
        return Card.np_cards[values - 3].tolist()


class CardGroup:
    def __init__(self, cards, t, val, len=1):
        self.type = t
        self.cards = cards
        self.value = val
        self.len = len

    def bigger_than(self, g):
        if self.type == Category.EMPTY:
            return g.type != Category.EMPTY
        if g.type == Category.EMPTY:
            return True
        if g.type == Category.BIGBANG:
            return False
        if self.type == Category.BIGBANG:
            return True
        if g.type == Category.QUADRIC:
            if self.type == Category.QUADRIC and self.value > g.value:
                return True
            else:
                return False
        if self.type == Category.QUADRIC or \
                (self.type == g.type and self.len == g.len and self.value > g.value):
            return True
        else:
            return False

    @staticmethod
    def isvalid(cards):
        return CardGroup.folks(cards) == 1

    @staticmethod
    def to_cardgroup(cards):
        candidates = CardGroup.analyze(cards)
        for c in candidates:
            if len(c.cards) == len(cards):
                return c
        print("cards error!")
        print(cards)
        raise Exception("Invalid Cards!")

    @staticmethod
    def folks(cards):
        cand = CardGroup.analyze(cards)
        cnt = 10000
        # if not cards:
        #     return 0
        # for c in cand:
        #     remain = list(cards)
        #     for card in c.cards:
        #         remain.remove(card)
        #     if CardGroup.folks(remain) + 1 < cnt:
        #         cnt = CardGroup.folks(remain) + 1
        # return cnt
        spec = False
        for c in cand:
            if c.type == Category.TRIPLE_LINE or c.type == Category.THREE_ONE or \
                    c.type == Category.THREE_TWO or c.type == Category.FOUR_TAKE_ONE or \
                    c.type == Category.FOUR_TAKE_TWO or c.type == Category.THREE_ONE_LINE or \
                    c.type == Category.THREE_TWO_LINE or c.type == Category.SINGLE_LINE or \
                    c.type == Category.DOUBLE_LINE:
                spec = True
                remain = list(cards)
                for card in c.cards:
                    remain.remove(card)
                if CardGroup.folks(remain) + 1 < cnt:
                    cnt = CardGroup.folks(remain) + 1
        if not spec:
            cnt = len(cand)
        return cnt

    @staticmethod
    def analyze(cards):
        cards = list(cards)
        if len(cards) == 0:
            return [CardGroup([], Category.EMPTY, 0)]
        candidates = []

        # TODO: this does not rule out Nuke kicker
        counts = Counter(cards)
        if '*' in cards and '$' in cards:
            candidates.append((CardGroup(['*', '$'], Category.BIGBANG, 100)))
            # cards.remove('*')
            # cards.remove('$')

        quadrics = []
        # quadric
        for c in counts:
            if counts[c] == 4:
                quadrics.append(c)
                candidates.append(CardGroup([c] * 4, Category.QUADRIC, Card.to_value(c)))
                cards = list(filter(lambda a: a != c, cards))

        counts = Counter(cards)
        singles = [c for c in counts if counts[c] == 1]
        doubles = [c for c in counts if counts[c] == 2]
        triples = [c for c in counts if counts[c] == 3]

        singles.sort(key=lambda k: Card.cards_to_value[k])
        doubles.sort(key=lambda k: Card.cards_to_value[k])
        triples.sort(key=lambda k: Card.cards_to_value[k])

        # continuous sequence
        if len(singles) > 0:
            cnt = 1
            cand = [singles[0]]
            for i in range(1, len(singles)):
                if Card.to_value(singles[i]) >= Card.to_value('2'):
                    break
                if Card.to_value(singles[i]) == Card.to_value(cand[-1]) + 1:
                    cand.append(singles[i])
                    cnt += 1
                else:
                    if cnt >= 5:
                        candidates.append(CardGroup(cand, Category.SINGLE_LINE, Card.to_value(cand[0]), cnt))
                        # for c in cand:
                        #     cards.remove(c)
                    cand = [singles[i]]
                    cnt = 1
            if cnt >= 5:
                candidates.append(CardGroup(cand, Category.SINGLE_LINE, Card.to_value(cand[0]), cnt))
                # for c in cand:
                #     cards.remove(c)

        if len(doubles) > 0:
            cnt = 1
            cand = [doubles[0]] * 2
            for i in range(1, len(doubles)):
                if Card.to_value(doubles[i]) >= Card.to_value('2'):
                    break
                if Card.to_value(doubles[i]) == Card.to_value(cand[-1]) + 1:
                    cand += [doubles[i]] * 2
                    cnt += 1
                else:
                    if cnt >= 3:
                        candidates.append(CardGroup(cand, Category.DOUBLE_LINE, Card.to_value(cand[0]), cnt))
                        # for c in cand:
                        # if c in cards:
                        #     cards.remove(c)
                    cand = [doubles[i]] * 2
                    cnt = 1
            if cnt >= 3:
                candidates.append(CardGroup(cand, Category.DOUBLE_LINE, Card.to_value(cand[0]), cnt))
                # for c in cand:
                # if c in cards:
                #     cards.remove(c)

        if len(triples) > 0:
            cnt = 1
            cand = [triples[0]] * 3
            for i in range(1, len(triples)):
                if Card.to_value(triples[i]) >= Card.to_value('2'):
                    break
                if Card.to_value(triples[i]) == Card.to_value(cand[-1]) + 1:
                    cand += [triples[i]] * 3
                    cnt += 1
                else:
                    if cnt >= 2:
                        candidates.append(CardGroup(cand, Category.TRIPLE_LINE, Card.to_value(cand[0]), cnt))
                        # for c in cand:
                        #     if c in cards:
                        #         cards.remove(c)
                    cand = [triples[i]] * 3
                    cnt = 1
            if cnt >= 2:
                candidates.append(CardGroup(cand, Category.TRIPLE_LINE, Card.to_value(cand[0]), cnt))
                # for c in cand:
                #     if c in cards:
                #         cards.remove(c)

        for t in triples:
            candidates.append(CardGroup([t] * 3, Category.TRIPLE, Card.to_value(t)))

        counts = Counter(cards)
        singles = [c for c in counts if counts[c] == 1]
        doubles = [c for c in counts if counts[c] == 2]

        # single
        for s in singles:
            candidates.append(CardGroup([s], Category.SINGLE, Card.to_value(s)))

        # double
        for d in doubles:
            candidates.append(CardGroup([d] * 2, Category.DOUBLE, Card.to_value(d)))

        # 3 + 1, 3 + 2
        for c in triples:
            triple = [c] * 3
            for s in singles:
                if s not in triple:
                    candidates.append(CardGroup(triple + [s], Category.THREE_ONE,
                                                Card.to_value(c)))
            for d in doubles:
                if d not in triple:
                    candidates.append(CardGroup(triple + [d] * 2, Category.THREE_TWO,
                                                Card.to_value(c)))

        # 4 + 2
        for c in quadrics:
            for extra in list(itertools.combinations(singles, 2)):
                candidates.append(CardGroup([c] * 4 + list(extra), Category.FOUR_TAKE_ONE,
                                            Card.to_value(c)))
            for extra in list(itertools.combinations(doubles, 2)):
                candidates.append(CardGroup([c] * 4 + list(extra) * 2, Category.FOUR_TAKE_TWO,
                                            Card.to_value(c)))
        # 3 * n + n, 3 * n + 2 * n
        triple_seq = [c.cards for c in candidates if c.type == Category.TRIPLE_LINE]
        for cand in triple_seq:
            cnt = int(len(cand) / 3)
            for extra in list(itertools.combinations(singles, cnt)):
                candidates.append(
                    CardGroup(cand + list(extra), Category.THREE_ONE_LINE,
                              Card.to_value(cand[0]), cnt))
            for extra in list(itertools.combinations(doubles, cnt)):
                candidates.append(
                    CardGroup(cand + list(extra) * 2, Category.THREE_TWO_LINE,
                              Card.to_value(cand[0]), cnt))

        importance = [Category.EMPTY, Category.SINGLE, Category.DOUBLE, Category.DOUBLE_LINE, Category.SINGLE_LINE,
                      Category.THREE_ONE,
                      Category.THREE_TWO, Category.THREE_ONE_LINE, Category.THREE_TWO_LINE,
                      Category.TRIPLE_LINE, Category.TRIPLE, Category.FOUR_TAKE_ONE, Category.FOUR_TAKE_TWO,
                      Category.QUADRIC, Category.BIGBANG]
        candidates.sort(key=functools.cmp_to_key(lambda x, y: importance.index(x.type) - importance.index(y.type)
        if importance.index(x.type) != importance.index(y.type) else x.value - y.value))
        # for c in candidates:
        #     print c.cards
        return candidates

def to_value(cards):
    if isinstance(cards, list) or isinstance(cards, np.ndarray):
        values = [Card.cards.index(c)+3 for c in cards]
        return values
    else:
        return Card.cards.index(cards)+3


# map 3 - 17 to char cards
def to_char(cards):
    if isinstance(cards, list) or isinstance(cards, np.ndarray):
        if len(cards) == 0:
            return []
        chars = [Card.cards[c-3] for c in cards]
        return chars
    else:
        return Card.cards[cards-3]
def char2ccardgroup(chars):
    cg = CardGroup.to_cardgroup(chars)
    ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
    return ccg

def ccardgroup2char(cg):
    return [to_char(int(c) + 3) for c in cg.cards]

def valid_actions(self_hands,last_hands):
    cself_hands = [CCard(to_value(c) - 3) for c in self_hands]
    # clast_hands = [CCard(to_value(c) - 3) for c in last_hands]
    clast_hands = char2ccardgroup(last_hands)
    actions = getActionspace(cself_hands, clast_hands,0,0)
    if len(actions) == 1:
        print('error')
    res = []
    for action in actions:
        caction = ccardgroup2char(action)
        if len(caction) == 0:
            res.append('EMPTY')
        else:
            res.append(caction)
    return res
    # return [ccardgroup2char(action) for action in actions]

def arr2cards(arr):
    """
    :param arr: 15 * 4
    :return: ['A','A','A', '3', '3'] 用 [3,3,14,14,14]表示
        [3,4,5,6,7,8,9,10, J, Q, K, A, 2,BJ,CJ]
        [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    """
    res = []
    for idx in range(15):
        for _ in range(arr[idx]):
            res.append(idx + 3)
    return np.array(res, dtype=np.int)

def cards2arr(cards):
    arr = np.zeros((15,), dtype=np.int)
    for card in cards:
        arr[card - 3] += 1
    return arr

def arr2onehot(arr,mask=0):
    res = np.zeros((1,15, 4), dtype=np.float32)
    for card_idx, count in enumerate(arr):
        if count > 0:
            res[0][card_idx][:int(count)] = 1
            if mask > 0:
                # if random.random() < mask:
                #     res[0][card_idx][:int(count)] = 0
                for i in range(0,count):
                    if random.random() < mask:
                        res[0][card_idx][i:int(i+1)] = 0


    return res

def batch_arr2onehot(batch_arr):
    res = np.zeros((len(batch_arr), 15, 4), dtype=np.int)
    for idx, arr in enumerate(batch_arr):
        if arr == 'EMPTY':
            continue
        cards = [i if not i == '10' else 'X' for i in arr]
        cards = [cfg.str_cards_dict[i] for i in cards]
        cards = cards2arr(cards)
        for card_idx, count in enumerate(cards):
            if count > 0:
                res[idx][card_idx][:int(count)] = 1
    return res

def onehot2arr(onehot_cards):
    """
    :param onehot_cards: 15 * 4
    :return: (15,)
    """
    res = np.zeros((15,), dtype=np.int)
    for idx, onehot in enumerate(onehot_cards):
        res[idx] = sum(onehot)
    return res

def cards2str(cards):
    res = [cfg.cards_str_dict[i] for i in cards]
    return res

def str2cards(str):
    str = str.replace('10','X')
    res = [cfg.str_cards_dict[i] for i in str]
    return res

def empty2onehot():
    return np.zeros((1,15, 4), dtype=np.int)

def str2arr(str):
    if str == 'EMPTY':
        return 'EMPTY'
    str = str.replace('10','X')
    res = []
    for s in str:
        if s == 'X':
            res.append('10')
        else:
            res.append(s)
    return res

def str2onehot(state,mask=0):
    if state == 'EMPTY':
        return empty2onehot()
    cards = str2cards(state)
    cards = cards2arr(cards)
    onehot = arr2onehot(cards,mask)
    return onehot

def oppHandsCount2onehot(state):
    count = 0 if state == 'EMPTY'else len(str2cards(state))
    assert count > 0
    res_init = np.zeros((15*4, 1), dtype=np.float32)
    res_init[:count] = 1
    res_init = res_init.reshape((15,4))
    res = np.zeros((1,15,4),dtype=np.float32)
    res[0] = res_init
    return res


def label2onehot(count,index):
    res = np.zeros((count, 1), dtype=np.float32)
    res[index][0] = 1
    return res