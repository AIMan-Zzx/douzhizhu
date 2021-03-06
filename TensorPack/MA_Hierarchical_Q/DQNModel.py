
import abc
from tensorpack import *
import tensorflow as tf
import tensorpack
from tensorpack import ModelDesc
from tensorpack.utils import logger
from tensorpack.tfutils import (
    varreplace, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import numpy as np
from tensorflow.contrib.layers import l2_regularizer
from conditional import conditional


def res_fc_block(inputs, units, stack=3):
    residual = inputs
    for i in range(stack):
        residual = FullyConnected('fc%d' % i, residual, units, activation=tf.nn.relu)
    x = inputs
    # x = FullyConnected('fc', x, units, activation=tf.nn.relu)
    if inputs.shape[1].value != units:
        x = FullyConnected('fc', x, units, activation=tf.nn.relu)
    return tf.contrib.layers.layer_norm(residual + x, scale=False)


class Model(ModelDesc):
    learning_rate = 1e-4
    cost_weights = [1., 1., 1.]

    def __init__(self, agent_names, state_shape, method, num_actions, gamma):
        self.agent_names = agent_names
        self.state_shape = state_shape
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma

    def inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        res = []
        for name in self.agent_names:
            res.extend([tf.placeholder(tf.float32,
                               (None, 2, self.state_shape[0], self.state_shape[1], self.state_shape[2]),
                               name + '_joint_state'),
                tf.placeholder(tf.int64, (None,), name + '_action'),
                tf.placeholder(tf.float32, (None,), name + '_reward'),
                tf.placeholder(tf.bool, (None,), name + '_isOver'),
                tf.placeholder(tf.bool, (None,), name + '_comb_mask'),
                tf.placeholder(tf.bool, (None, 2, None), name + '_joint_fine_mask')])
        return res

    # input B * C * D
    # output B * C * 1
    def _get_DQN_prediction_comb(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        units = [512, 256, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, 1)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, 1)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.reshape(Q, [-1, shape[1], 1])

    # input B * N * D
    # output B * N * 1
    def _get_DQN_prediction_fine(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        units = [512, 256, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, 1)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, 1)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.reshape(Q, [-1, shape[1], 1])

    def _get_global_feature(self, joint_state):
        shape = joint_state.shape.as_list()
        net = tf.reshape(joint_state, [-1, shape[-1]])
        units = [256, 512, 1024]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        net = tf.reshape(net, [-1, shape[1], shape[2], units[-1]])
        return tf.reduce_max(net, [2])

    # decorate the function
    # output : B * A
    @auto_reuse_variable_scope
    def get_DQN_prediction(self, joint_state, comb_mask, fine_mask):
        with tensorpack.argscope([tensorpack.FullyConnected], kernel_initializer=tf.contrib.layers.xavier_initializer()):
            batch_size = tf.shape(joint_state)[0]
            with tf.variable_scope('dqn_global'):
                global_feature = self.get_global_feature(joint_state)

            comb_mask_idx = tf.cast(tf.where(comb_mask), tf.int32)
            with tf.variable_scope('dqn_comb'):
                q_comb = self._get_DQN_prediction_comb(tf.gather(global_feature, comb_mask_idx[:, 0]))
            q_comb = tf.squeeze(q_comb, -1)
            q_comb = tf.scatter_nd(comb_mask_idx, q_comb, tf.stack([batch_size, q_comb.shape[1]]))

            fine_mask_idx = tf.cast(tf.where(tf.logical_not(comb_mask)), tf.int32)
            state_fine = tf.concat([tf.tile(tf.expand_dims(global_feature, 2), [1, 1, joint_state.shape.as_list()[2], 1]), joint_state], -1)
            state_fine = tf.gather(state_fine[:, 0, :, :], fine_mask_idx[:, 0])
            with tf.variable_scope('dqn_fine'):
                q_fine = self._get_DQN_prediction_fine(state_fine)
            q_fine = tf.squeeze(q_fine, -1)
            q_fine = tf.scatter_nd(fine_mask_idx, q_fine, tf.stack([batch_size, q_fine.shape[1]]))

            larger_dim = max(joint_state.shape.as_list()[1], joint_state.shape.as_list()[2])
            padding_np = np.zeros([1, larger_dim], dtype=np.float32)
            padding_np[0, min(joint_state.shape[1], joint_state.shape[2]):] = -1e5
            padding = tf.convert_to_tensor(padding_np)
            # padding = tf.Variable(initial_value=padding_np, trainable=False, name='padding')
            padding = tf.tile(padding, tf.stack(
                [tf.shape(fine_mask_idx if joint_state.shape[1] > joint_state.shape[2] else comb_mask_idx)[0], 1]))
            padding = tf.scatter_nd(fine_mask_idx if joint_state.shape[1] > joint_state.shape[2] else comb_mask_idx,
                                    padding, tf.stack([batch_size, larger_dim]))
            # padding = tf.Print(padding, [padding], summarize=100)
            q = tf.add(tf.pad(q_comb, [[0, 0], [0, larger_dim - q_comb.shape.as_list()[1]]]) + tf.pad(q_fine, [[0, 0], [0, larger_dim - q_fine.shape.as_list()[ 1]]]),
                          padding)

            # q[tf.where(tf.logical_not(fine_mask))] = -1e5
            return tf.add(q, -tf.cast(tf.logical_not(fine_mask), dtype=tf.float32) * 1e5, name='Qvalue')

    # input :B * COMB * N * D
    # output : B * COMB * D'
    @auto_reuse_variable_scope
    def get_global_feature(self, joint_state):
        return self._get_global_feature(joint_state)

    # joint state: B * 2 * COMB * N * D for now, D = 256
    # dynamic action range
    def build_graph(self, *args):
        costs = []
        for i, name in enumerate(self.agent_names):
            joint_state, action, reward, isOver, comb_mask, joint_fine_mask = args[i * 6:(i + 1) * 6]
            with tf.variable_scope(name):
                with conditional(name is None, varreplace.freeze_variables()):
                    state = tf.identity(joint_state[:, 0, :, :, :], name='state')
                    fine_mask = tf.identity(joint_fine_mask[:, 0, :], name='fine_mask')
                    self.predict_value = self.get_DQN_prediction(state, comb_mask, fine_mask)
                    if not get_current_tower_context().is_training:
                        continue

                    # reward = tf.clip_by_value(reward, -1, 1)
                    next_state = tf.identity(joint_state[:, 1, :, :, :], name='next_state')
                    next_fine_mask = tf.identity(joint_fine_mask[:, 1, :], name='next_fine_mask')
                    action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

                    pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
                    max_pred_reward = tf.reduce_mean(tf.reduce_max(
                        self.predict_value, 1), name='predict_reward')
                    summary.add_moving_summary(max_pred_reward)

                    with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
                        # we are alternating between comb and fine states
                        targetQ_predict_value = self.get_DQN_prediction(next_state, tf.logical_not(comb_mask), next_fine_mask)    # NxA

                    if self.method != 'Double':
                        # DQN
                        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
                    else:
                        # Double-DQN
                        next_predict_value = self.get_DQN_prediction(next_state, tf.logical_not(comb_mask), next_fine_mask)
                        self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
                        predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
                        best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

                    target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
                    # target = tf.Print(target, [target], summarize=100)
                    # tf.assert_greater(target, -100., message='target error')
                    # tf.assert_greater(pred_action_value, -100., message='pred value error')
                    # pred_action_value = tf.Print(pred_action_value, [pred_action_value], summarize=100)

                    l2_loss = tensorpack.regularize_cost(name + '/dqn.*W{1}', l2_regularizer(1e-3))
                    # cost = tf.losses.mean_squared_error(target, pred_action_value)
                    with tf.control_dependencies([tf.assert_greater(target, -100., message='target error'), tf.assert_greater(pred_action_value, -100., message='pred value error')]):
                        cost = tf.losses.huber_loss(
                                        target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
                    summary.add_param_summary((name + '.*/W', ['histogram', 'rms']))   # monitor all W
                    summary.add_moving_summary(cost)
                    costs.append(cost)
        if not get_current_tower_context().is_training:
            return
        #zzx return tf.add_n([costs[i] * self.cost_weights[i] for i in range(3)])
        return tf.add_n([costs[i] * self.cost_weights[i] for i in range(2)])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        # opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [
                # gradproc.GlobalNormClip(2.0),
                gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])

    # @staticmethod
    def update_target_param(self):
        # update target network together
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            for name in self.agent_names:
                if target_name.startswith(name + '/target'):
                    new_name = target_name.replace('target/', '')
                    logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
                    ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network_all')