import tensorflow as tf
import numpy as np
from networks.deep_feudal.batch_processor import DeepFeudalBatchProcessor
from networks.deep_feudal.citizens import SuperManager, Manager, Worker
import common
from networks import util

from pysc2.lib import actions


class DeepFeudalPolicy:
    def __init__(self, obs_spec, config):
        self.obs_spec = obs_spec
        self.config = config
        self.s_size = obs_spec["screen"][1]
        self.m_size = obs_spec["minimap"][1]
        self.num_actions = len(actions.FUNCTIONS)

        self.worker = Worker(config, self.num_actions, self.s_size, self.m_size)
        self.manager = Manager(config)
        self.super_manager = SuperManager(config)

        self.epsilon = config.epsilon_start

        self.scope = "Deep_FeUdal_Policy"

        self.state_in = []
        self.state_out = []

        self.pi = None
        self.obs = None
        self.actions = None
        self.pi_loss = None
        self.vf_loss = None
        self.loss = None
        self.var_list = None

        self.deep_batch_processor = DeepFeudalBatchProcessor(self.super_manager.c)
        self.batch_processor = DeepFeudalBatchProcessor(self.manager.c)

    def build_placeholders(self):
        self.obs = common.get_feature_placeholders(self.s_size, self.m_size)
        self.actions = {
            "spatial": tf.placeholder(tf.int32, [None], "spatial_actions"),
            "non_spatial": tf.placeholder(tf.int32, [None], "non_spatial_actions")
        }

        self.super_manager.build_placeholders()
        self.manager.build_placeholders()
        self.worker.build_placeholders()

    def build_model(self):
        with tf.variable_scope(self.scope):
            self.worker.build_perception(self.obs)
            self.manager.build_perception(self.worker.s_fc)
            self.super_manager.build_perception(self.worker.s_fc)
            self.super_manager.build_model(self.state_in, self.state_out)
            self.manager.build_model(self.super_manager.get_w, self.state_in, self.state_out)
            self.worker.build_model(self.manager.get_w, self.obs, self.state_in, self.state_out)
            self.pi = self.worker.pi

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        if not tf.get_variable_scope().name.startswith("global"):
            with tf.name_scope("loss"):
                self.super_manager.build_loss()
                self.manager.build_loss()
                self.worker.build_loss(self.actions)
                self.worker.build_entropy()

                # loss
                self.worker.loss = self.worker.pi_loss + self.worker.vf_loss - self.worker.entropy
                self.manager.loss = self.manager.pi_loss + self.manager.vf_loss
                self.super_manager.loss = self.super_manager.pi_loss + self.super_manager.vf_loss
                # self.pi_loss = self.worker.loss + self.manager.loss + self.super_manager.loss
                # self.vf_loss = self.worker.vf_loss + self.manager.vf_loss + self.super_manager.vf_loss
                # self.loss = self.pi_loss + self.vf_loss - self.worker.entropy
                # self.loss = self.worker.loss + self.manager.loss + self.super_manager.loss

    def get_initial_features(self):
        return (
            np.zeros((1, self.super_manager.c - 1, self.super_manager.g_dim), np.float32),
            0,
            np.zeros((1, self.manager.c - 1, self.manager.g_dim), np.float32),
            0,
            self.super_manager.state_init + self.manager.state_init + self.worker.lstm.state_init
        )

    def act(self, obs, sm_gp, sm_idx, m_gp, m_idx, last_features):
        sess = tf.get_default_session()

        # super manager sess
        feed_dict_1 = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_1.update({
            self.state_in[0]: last_features[0], self.state_in[1]: last_features[1],
            self.super_manager.dilated_idx_in: sm_idx
        })

        fetches_1 = [
            self.super_manager.g, self.super_manager.s,
            self.super_manager.dilated_idx_out,
            self.super_manager.vf,
            self.state_out[0], self.state_out[1]
        ]

        # g, s, idx, vf, co, ho
        super_manager_fetched = sess.run(fetches_1, feed_dict_1)

        # manager sess
        feed_dict_2 = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_2.update({
            self.super_manager.g_in: super_manager_fetched[0],
            self.state_in[2]: last_features[2], self.state_in[3]: last_features[3],
            self.manager.dilated_idx_in: m_idx,
            self.super_manager.prev_g: sm_gp
        })

        fetches_2 = [
            self.manager.g, self.manager.s,
            self.manager.dilated_idx_out,
            self.manager.vf,
            self.super_manager.last_c_g,
            self.state_out[2], self.state_out[3]
        ]

        # g, s, idx, vf, last_c_g, co, ho
        manager_fetched = sess.run(fetches_2, feed_dict_2)

        # worker sess
        feed_dict_3 = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_3.update({
            self.manager.g_in: manager_fetched[0],
            self.state_in[4]: last_features[4], self.state_in[5]: last_features[5],
            self.manager.prev_g: m_gp
        })

        fetches_3 = [
            self.pi,
            self.manager.last_c_g,
            self.state_out[4], self.state_out[5]
        ]

        # pi, last_c_g, co, ho
        worker_fetched = sess.run(fetches_3, feed_dict_3)
        valid_actions = obs["available_actions"]
        if self.config.e_greedy:
            if np.random.random() < self.epsilon:
                act = util.exploit_distribution(worker_fetched[0], valid_actions, self.s_size)
            else:
                act = util.exploit_max(worker_fetched[0], valid_actions, self.s_size)
        else:
            act = util.exploit_distribution(worker_fetched[0], valid_actions, self.s_size)

        features = super_manager_fetched[-2:] + manager_fetched[-2:] + worker_fetched[-2:]
        return act, super_manager_fetched[:-2], manager_fetched[:-2], worker_fetched[:-2], features

    def value(self, obs, sm_gp, sm_idx, m_gp, m_idx, last_features):
        sess = tf.get_default_session()

        # super manager sess
        feed_dict_1 = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_1.update({
            self.state_in[0]: last_features[0], self.state_in[1]: last_features[1],
            self.super_manager.dilated_idx_in: sm_idx
        })

        sm_vf, sm_g = sess.run([self.super_manager.vf, self.super_manager.g], feed_dict_1)

        # manager sess
        feed_dict_2 = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_2.update({
            self.super_manager.g_in: sm_g,
            self.state_in[2]: last_features[2], self.state_in[3]: last_features[3],
            self.manager.dilated_idx_in: m_idx,
            self.super_manager.prev_g: sm_gp
        })

        m_vf, m_g = sess.run([self.manager.vf, self.manager.g], feed_dict_2)

        # worker sess
        feed_dict_3 = common.populate_feature_dict(obs, self.obs, self.s_size)
        feed_dict_3.update({
            self.manager.g_in: m_g,
            self.state_in[4]: last_features[4], self.state_in[5]: last_features[5],
            self.manager.prev_g: m_gp
        })

        w_vf = sess.run(self.worker.vf, feed_dict_3)

        return sm_vf[0], m_vf[0], w_vf[0]

    def update_deep_batch(self, batch):
        return self.deep_batch_processor.process_batch(batch)

    def update_batch(self, batch):
        return self.batch_processor.process_batch(batch)
