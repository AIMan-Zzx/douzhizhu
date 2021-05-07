import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from simulator.tools import *
import ctypes
import threading

from simulator.predictor import Predictor
from card import Card
from simulator.coordinator import Coordinator
import multiprocessing
import zmq
from tensorpack.utils.serialize import dumps, loads
from env import Env as CEnv
from TensorPack.MA_Hierarchical_Q.DQNModel import Model
from TensorPack.MA_Hierarchical_Q.env import Env
from simulator.expreplay import ExpReplay
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
import six
from abc import abstractmethod, ABCMeta
from tensorpack import *
from tensorpack.utils.stats import StatCounter
import argparse
from simulator.sim import Simulator
from simulator.manager import SimulatorManager

toggle = multiprocessing.Value('i', 0)
toggle.value = 1


BATCH_SIZE = 8
MAX_NUM_COMBS = 100
MAX_NUM_GROUPS = 21
ATTEN_STATE_SHAPE = 60

# HIDDEN_STATE_DIM = 256 + 256 * 2 + 120
HIDDEN_STATE_DIM = 256 + 256 + 60
STATE_SHAPE = (MAX_NUM_COMBS, MAX_NUM_GROUPS, HIDDEN_STATE_DIM)
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1000
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = cf_offline.steps_per_epoch
EVAL_EPISODE = 100

NUM_ACTIONS = None
METHOD = None


def esc_pressed():
    global toggle
    toggle.value = 1 - toggle.value
    print("toggle changed to ", toggle.value)


def hook():
    ctypes.windll.user32.RegisterHotKey(None, 1, 0, win32con.VK_ESCAPE)

    try:
        msg = ctypes.wintypes.MSG()
        while ctypes.windll.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == win32con.WM_HOTKEY:
                esc_pressed()
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageA(ctypes.byref(msg))
    finally:
        ctypes.windll.user32.UnregisterHotKey(None, 1)


class MyDataFLow(DataFlow):
    def __init__(self, exps):
        self.exps = exps

    def get_data(self):
        gens = [e.get_data() for e in self.exps]
        while True:
            batch = []
            # TODO: balance between batch generation?
            for g in gens:
                batch.extend(next(g))
            yield batch


if __name__ == '__main__':
    window_name = cf_offline.window_name

    def callback(h, extra):
        if window_name in win32gui.GetWindowText(h):
            extra.append(h)
        return True


    hwnds = []
    win32gui.EnumWindows(callback, hwnds)

    name_sim2exp = 'tcp://127.0.0.1:1234'
    name_exp2sim = 'tcp://127.0.0.1:2234'

    name_sim2coord = 'tcp://127.0.0.1:3234'
    name_coord2sim = 'tcp://127.0.0.1:4234'

    name_sim2mgr = 'tcp://127.0.0.1:5234'
    name_mgr2sim = 'tcp://127.0.0.1:6234'

    agent_names = ['agent%d' % i for i in range(1, 4)]

    rect = get_window_rect(hwnds[0])
    print(hwnds)

    # no exploration now
    sims = [Simulator(idx=i, hwnd=hwnds[i], pipe_sim2exps=[name_sim2exp + str(j) for j in range(3)], pipe_exps2sim=[name_exp2sim + str(j) for j in range(3)],
                      pipe_sim2coord=name_sim2coord, pipe_coord2sim=name_coord2sim,
                      pipe_sim2mgr=name_sim2mgr, pipe_mgr2sim=name_mgr2sim,
                      agent_names=agent_names, exploration=1.0, toggle=toggle) for i in range(len(hwnds))]

    manager = SimulatorManager(sims, pipe_sim2mgr=name_sim2mgr, pipe_mgr2sim=name_mgr2sim)

    coordinator = Coordinator(agent_names, sim2coord=name_sim2coord, coord2sim=name_coord2sim)

    # coordinator._before_train()
    # sim.start()
    # sim.join()
    hotkey_t = threading.Thread(target=hook, args=())

    def get_config():

        model = Model(agent_names, STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA)
        exps = [ExpReplay(
            agent_name=name,
            state_shape=STATE_SHAPE,
            num_actions=[MAX_NUM_COMBS, MAX_NUM_GROUPS],
            batch_size=BATCH_SIZE,
            memory_size=MEMORY_SIZE,
            init_memory_size=INIT_MEMORY_SIZE,
            init_exploration=1.,
            update_frequency=UPDATE_FREQ,
            pipe_exp2sim=name_exp2sim + str(i),
            pipe_sim2exp=name_sim2exp + str(i)
        ) for i, name in enumerate(agent_names)]

        df = MyDataFLow(exps)

        return AutoResumeTrainConfig(
            # always_resume=False,
            data=QueueInput(df),
            model=model,
            callbacks=[
                ModelSaver(),
                PeriodicTrigger(
                    RunOp(model.update_target_param, verbose=True),
                    every_k_steps=STEPS_PER_EPOCH // 10),  # update target network every 10k steps
                # the following order is important

                coordinator,
                manager,
                *exps,
                # ScheduledHyperParamSetter('learning_rate',
                #                           [(60, 5e-5), (100, 2e-5)]),
                *[ScheduledHyperParamSetter(
                    ObjAttrParam(sim, 'exploration'),
                    [(0, 1), (30, 0.5), (100, 0.3), (320, 0.1)],  # 1->0.1 in the first million steps
                    interp='linear') for sim in sims],
                # Evaluator(EVAL_EPISODE, agent_names, lambda: Env(agent_names)),
                HumanHyperParamSetter('learning_rate'),
            ],
            session_init=ChainInit(
                [SaverRestore('../TensorPack/MA_Hierarchical_Q/train_log/DQN-60-MA/model-355000')]),
            # starting_epoch=0,k
            # session_init=SaverRestore('train_log/DQN-54-AUG-STATE/model-75000'),
            steps_per_epoch=STEPS_PER_EPOCH,
            max_epoch=1000,
        )
    # client = RandomClient(pipe_c2s=namec2s, pipe_s2c=names2c)
    # # client.run()
    # ensure_proc_terminate(client)
    # # start_proc_mask_signal(client)
    # client.start()
    # client.join()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo
    # set num_actions
    NUM_ACTIONS = max(MAX_NUM_GROUPS, MAX_NUM_COMBS)

    nr_gpu = get_nr_gpu()
    train_tower = list(range(nr_gpu))
    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state', 'comb_mask'],
            output_names=['Qvalue']))
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-REALDATA'))
        config = get_config()
        hotkey_t.start()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(config, trainer)

        hotkey_t.join()