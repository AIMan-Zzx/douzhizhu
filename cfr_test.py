import numpy as np
poker = [1,2,3]
in_game_state = ['1','1b','1p','1pb','2','2b','2p','2pb','3','3b','3p','3pb']

states = {'1': [0.5, 0.5], '1b': [0.5, 0.5], '1p': [0.5, 0.5], '1pb': [0.5, 0.5], '2': [0.5, 0.5], '2b': [0.5, 0.5],
          '2p': [0.5, 0.5], '2pb': [0.5, 0.5], '3': [0.5, 0.5], '3b': [0.5, 0.5], '3p': [0.5, 0.5], '3pb': [0.5, 0.5]}

end = ['pp', 'pbp', 'pbb', 'bp', 'bb']

CFR_r = {'1': [0, 0], '1b': [0, 0], '1p': [0, 0], '1pb': [0, 0], '2': [0, 0], '2b': [0, 0],
         '2p': [0, 0], '2pb': [0, 0], '3': [0, 0], '3b': [0, 0], '3p': [0, 0], '3pb': [0, 0]}

pass_bet = {'p': 0, 'b': 1}

action = ['p', 'b']

# CFR中间状态
CFR_s = {'1': [0, 0], '1b': [0, 0], '1p': [0, 0], '1pb': [0, 0], '2': [0, 0], '2b': [0, 0],
         '2p': [0, 0], '2pb': [0, 0], '3': [0, 0], '3b': [0, 0], '3p': [0, 0], '3pb': [0, 0]}
CFR_s2 = {'1': [0, 0], '1b': [0, 0], '1p': [0, 0], '1pb': [0, 0], '2': [0, 0], '2b': [0, 0],
          '2p': [0, 0], '2pb': [0, 0], '3': [0, 0], '3b': [0, 0], '3p': [0, 0], '3pb': [0, 0]}

# 已经计算好的策略
AI_state = {'1': [0.8574384003665548, 0.14256159963344525], '1b': [0.999985067494923, 1.4932505077051726e-05],
            '1p': [0.660594597675478, 0.33940540232452193], '1pb': [0.9999911732218897, 8.826778110280645e-06],
            '2': [0.9997037628675545, 0.00029623713244554913], '2b': [0.6618884775698471, 0.3381115224301529],
            '2p': [0.9999699762812622, 3.0023718737802863e-05], '2pb': [0.5291090066614956, 0.4708909933385044],
            '3': [0.5781543746887431, 0.42184562531125686], '3b': [3.0112318949682314e-05, 0.9999698876810503],
            '3p': [1.5056159474841157e-05, 0.9999849438405252], '3pb': [1.2844892512449064e-05, 0.9999871551074876]}

AI_state2 = {'1': [0.9005897997728478, 0.0994102002271522], '1b': [0.9999998499623105, 1.5003768946759426e-07],
             '1p': [0.6660743153746168, 0.3339256846253833], '1pb': [0.999999916728828, 8.327117200450158e-08],
             '2': [0.9999943092803633, 5.6907196367392786e-06], '2b': [0.6667338506314542, 0.33326614936854576],
             '2p': [0.9999989492266308, 1.050773369199731e-06], '2pb': [0.5656811999441647, 0.43431880005583534],
             '3': [0.7008390070018635, 0.2991609929981364], '3b': [1.498520660404049e-07, 0.9999998501479339],
             '3p': [1.498520660404049e-07, 0.9999998501479339], '3pb': [1.0708730371989468e-07, 0.9999998929126963]}

AI_state3 = {'1': [0.8273490325307964, 0.17265096746920355], '1b': [0.999998499111474, 1.5008885260073963e-06],
             '1p': [0.6667986655056403, 0.3332013344943597], '1pb': [0.9999990919856274, 9.080143725515324e-07],
             '2': [0.9999405784977654, 5.94215022346119e-05], '2b': [0.6643358318791153, 0.3356641681208847],
             '2p': [0.9999801511805928, 1.98488194072019e-05], '2pb': [0.4941674200495377, 0.5058325799504624],
             '3': [0.4823765523726862, 0.5176234476273138], '3b': [1.5010912933702802e-06, 0.9999984989087066],
             '3p': [1.5010912933702802e-06, 0.9999984989087066], '3pb': [1.555232740630912e-06, 0.9999984447672594]}

AI_state4 = {'1': [0.834303037809882, 0.1656969621901179], '1b': [0.9999984955709609, 1.5044290390910841e-06],
             '1p': [0.6659521147790602, 0.33404788522093976], '1pb': [0.9999991023431777, 8.976568223041392e-07],
             '2': [0.9999750183896345, 2.4981610365392336e-05], '2b': [0.662709856181752, 0.33729014381824796],
             '2p': [0.9999927129173443, 7.287082655696929e-06], '2pb': [0.5039222434922419, 0.4960777565077581],
             '3': [0.5037166880570191, 0.49628331194298103], '3b': [1.5008164441456152e-06, 0.9999984991835559],
             '3p': [1.5008164441456152e-06, 0.9999984991835559], '3pb': [1.4874700238643287e-06, 0.9999985125299762]}


class Player(object):
    def __init__(self, is_human, hand, decision):
        self.is_human = is_human  # 是否人类玩家
        self.hand = hand  # 手牌
        self.information_set = str(hand)  # 所处信息集
        self.decision = decision  # 策略集上的概率分布

    # 选择出牌
    def choice(self, h, display):
        if self.is_human == 0:
            print('your hand is:', self.hand)
            return self.human_choice()
        else:
            return self.AI_choice(h, display)

    def human_choice(self):
        a = input('p or b?:')
        print('Human choice:', a)
        self.information_set += a
        return a

    def AI_choice(self, h, display):
        a = np.random.choice(['p', 'b'], p=self.decision[str(self.hand) + h])
        if display == 1:
            print('AI distribution:', self.decision[str(self.hand) + h])
            print('AI choice:', a)
        self.information_set += a
        return a


class Game(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.end = 0
        self.result = []

    # 游戏流程
    def flow(self, h, display):
        if h in end:
            self.end = 1
            self.result = self.judge(h)
            return
        if len(h) % 2 == 0:
            a = self.p1.choice(h, display=display)
        else:
            a = self.p2.choice(h, display=display)
        self.flow(h + a, display)

    # 判断胜负
    def judge(self, h):
        if h == 'pp':
            if self.p1.hand > self.p2.hand:
                return [1, -1]
            else:
                return [-1, 1]

        if h == 'pbp':
            return [-1, 1]

        if h == 'pbb' or h == 'bb':
            if self.p1.hand > self.p2.hand:
                return [2, -2]
            else:
                return [-2, 2]

        if h == 'bp':
            return [1, -1]

    # 显示结果
    def show_result(self):
        print('result:', self.result)
        print('p1 hand:', self.p1.hand)
        print('p2 hand:', self.p2.hand)

    # CFR流程
    def CFR_algorithm(self, h, pai1, pai2):
        if h in end:
            tmp = self.judge(h)
            return tmp

        if len(h) % 2 == 0:
            tmp_h = str(self.p1.hand) + h
        else:
            tmp_h = str(self.p2.hand) + h
        va = [0, 0]
        for a in action:
            if len(h) % 2 == 0:
                tmp_va = self.CFR_algorithm(h + a, pai1 * states[tmp_h][pass_bet[a]], pai2)
                va[pass_bet[a]] = tmp_va[0]
            else:
                tmp_va = self.CFR_algorithm(h + a, pai1, pai2 * states[tmp_h][pass_bet[a]])
                va[pass_bet[a]] = tmp_va[1]
        # 平均虚拟效用
        ave_va = states[tmp_h][0] * va[0] + states[tmp_h][1] * va[1]

        if len(h) % 2 == 0:
            oppo_pai = pai2
            self_pai = pai1
        else:
            oppo_pai = pai1
            self_pai = pai2

        CFR_r[tmp_h][0] = CFR_r[tmp_h][0] + oppo_pai * (va[0] - ave_va)
        CFR_r[tmp_h][1] = CFR_r[tmp_h][1] + oppo_pai * (va[1] - ave_va)
        CFR_s[tmp_h][0] = CFR_s[tmp_h][0] + self_pai * states[tmp_h][0]
        CFR_s[tmp_h][1] = CFR_s[tmp_h][1] + self_pai * states[tmp_h][1]
        CFR_s2[tmp_h][0] = CFR_s[tmp_h][0] / (CFR_s[tmp_h][0] + CFR_s[tmp_h][1])
        CFR_s2[tmp_h][1] = CFR_s[tmp_h][1] / (CFR_s[tmp_h][0] + CFR_s[tmp_h][1])

        if len(h) % 2 == 0:
            self.change_states(h, 1)
            return [ave_va, -ave_va]
        else:
            self.change_states(h, 2)
            return [-ave_va, ave_va]

    def change_states(self, h, p):
        if p == 1:
            tmp_h = str(self.p1.hand) + h
        else:
            tmp_h = str(self.p2.hand) + h

        p = max([CFR_r[tmp_h][0], 0])
        b = max([CFR_r[tmp_h][1], 0])
        if p == 0 and b == 0:
            states[tmp_h] = [0.5, 0.5]
        else:
            states[tmp_h] = [p / (p + b), b / (p + b)]



if __name__ == '__main__':
    '''
    np.random.seed(1)
    # 训练好的代码进行人机游戏
    np.random.shuffle(poker)
    # p1是人 p2代表电脑
    p1 = Player(0, poker[0], AI_state)
    p2 = Player(2, poker[1], AI_state)
    game1 = Game(p1, p2)
    game1.flow('', 1)
    game1.show_result()
    '''
    # 自己进行CFR训练

    p1 = Player(2, poker[0], states)
    p2 = Player(2, poker[1], states)
    game1 = Game(p1, p2)
    for i in range(100000):
        np.random.shuffle(poker)
        p1.hand = poker[0]
        p2.hand = poker[1]
        game1.CFR_algorithm('', 1, 1)
    print(CFR_s2)




