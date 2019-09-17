import numpy as np
import common

import tensorflow as tf

from pysc2.lib import actions

from networks.feudal.batch_processor import FeudalBatchProcessor
from networks import util
from networks.feudal.citizens import SuperManager, Worker

screen = common.SCREEN_KEY
minimap = common.MINIMAP_KEY


class FeudalPolicy:
    def __init__(self, obs_spec, config):
        self.obs_spec = obs_spec
        self.config = config

        self.num_actions = len(actions.FUNCTIONS)
        self.s_size = obs_spec[screen][1]
        self.m_size = obs_spec[minimap][1]

        self.worker = Worker(config, self.num_actions, self.s_size, self.m_size)
        self.manager = SuperManager(config, "manager")

        self.batch_processor = FeudalBatchProcessor(self.manager.c)

        self.epsilon = self.config.epsilon_start

        self.scope = "FeUdal_Policy"
        self.state_in = []
        self.state_out = []

        self.obs = None
        self.actions = None
        self.pi = None
        self.pi_loss = None
        self.vf_loss = None
        self.loss = None
        self.var_list = None

    def build_placeholders(self):
        self.obs = common.get_feature_placeholders(self.s_size, self.m_size)
        self.actions = {
            "spatial": tf.placeholder(tf.int32, [None], "spatial_actions"),
            "non_spatial": tf.placeholder(tf.int32, [None], "non_spatial_actions")
        }

        self.manager.build_placeholders()
        self.worker.build_placeholders()

    def build_model(self):
        with tf.variable_scope(self.scope):
            self.worker.build_perception(self.obs)  # z_t, (1) from 1703.01161
            self.manager.build_perception(self.worker.s_fc)  # s_t, (2) from 1703.01161
            self.manager.build_model(self.state_in, self.state_out)
            self.worker.build_model(self.manager.get_w, self.obs, self.state_in, self.state_out)
            self.pi = self.worker.pi

            if not tf.get_variable_scope().name.startswith("global"):
                with tf.name_scope("loss"):
                    self.manager.build_loss()
                    self.worker.build_loss(self.actions)
                    self.worker.build_entropy()

                    # loss
                    self.worker.loss = self.worker.pi_loss + self.worker.vf_loss - self.worker.entropy
                    self.manager.loss = self.manager.pi_loss + self.manager.vf_loss
                    # self.pi_loss = self.worker.loss + self.manager.loss
                    # self.vf_loss = self.worker.vf_loss + self.manager.vf_loss
                    # self.loss = self.pi_loss + self.vf_loss - self.worker.entropy
                    # self.pi_loss = self.worker.loss + self.manager.loss + self.super_manager.loss
                    # self.vf_loss = self.worker.vf_loss + self.manager.vf_loss + self.super_manager.vf_loss
                    # self.loss = self.pi_loss + self.vf_loss - self.worker.entropy
                    self.loss = self.worker.loss + self.manager.loss

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return np.zeros((1, self.config.c - 1, self.manager.g_dim), np.float32), \
               0, self.manager.state_init + self.worker.lstm.state_init

    def act(self, obs, gp, idx, cm, hm, cw, hw):
        sess = tf.get_default_session()
        obs_dict = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_1 = obs_dict
        feed_dict_1.update({
            self.state_in[0]: cm, self.state_in[1]: hm,
            self.manager.dilated_idx_in: idx
        })

        fetches_1 = [
            self.manager.g,
            self.manager.s,
            self.manager.dilated_idx_out,
            self.manager.vf,
            self.state_out[0],
            self.state_out[1]
        ]

        g, s, idx, vf, cmo, hmo = sess.run(fetches_1, feed_dict_1)

        feed_dict_2 = obs_dict
        feed_dict_2.update({
            self.manager.g_in: g,
            self.state_in[2]: cw, self.state_in[3]: hw,
            self.manager.prev_g: gp
        })

        fetches_2 = [
            self.worker.pi,
            self.manager.last_c_g,
            self.state_out[2],
            self.state_out[3]
        ]

        pi, last_c_g, cwo, hwo = sess.run(fetches_2, feed_dict_2)
        valid_actions = obs["available_actions"]
        if self.config.e_greedy:
            if np.random.random() < self.epsilon:
                act = util.exploit_distribution(pi, valid_actions, self.s_size)
            else:
                act = util.exploit_max(pi, valid_actions, self.s_size)
        else:
            act = util.exploit_distribution(pi, valid_actions, self.s_size)

        return act, vf, g, s, last_c_g, idx, cmo, hmo, cwo, hwo

    def value(self, obs, gp, idx, cm, hm, cw, hw):
        sess = tf.get_default_session()
        obs_dict = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_1 = obs_dict
        feed_dict_1.update({
            self.state_in[0]: cm, self.state_in[1]: hm,
            self.manager.dilated_idx_in: idx
        })

        manager_vf, g = sess.run([self.manager.vf, self.manager.g], feed_dict_1)

        feed_dict_2 = obs_dict
        feed_dict_2.update({
            self.manager.g_in: g,
            self.state_in[2]: cw, self.state_in[3]: hw,
            self.manager.prev_g: gp
        })

        worker_vf = sess.run(self.worker.vf, feed_dict_2)

        return manager_vf[0], worker_vf[0]

    def update_batch(self, batch):
        return self.batch_processor.process_batch(batch)
