import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
#from tensorpack.utils.stats import StatCounter
#from tensorpack.utils.utils import get_tqdm
from multiprocessing import *
from datetime import datetime
from scripts.envs import make_env#from envs import make_env
from scripts.agents import make_agent

types = ['RHCP']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def eval_episode(env, agent):
    env.reset()
    env.prepare()
    done = False
    r = 0
    while not done:
        if env.get_role_ID() != agent.role_id:

            #手动
            #r, done = env.manual_step_auto()

            #机器
            r, done = env.step_auto()

        else:

            #机器
            r, done = env.step(agent.intention(env))


            #手动
            #r, done = env.step(agent.manual_intention(env))


    if agent.role_id == 1:
        r = -r
    assert r != 0
    return int(r > 0)


def eval_proc(file_name):
    agent = make_agent('DEEPNET',1)
    # agent = make_agent('MCT',1)
    env = make_env('MCT')
    total = 1
    for i in range(total):
        winning_rate = eval_episode(env, agent)
        print('total {} ,finish {}'.format(total,i))


if __name__ == '__main__':
    eval_proc('res%d.txt')




