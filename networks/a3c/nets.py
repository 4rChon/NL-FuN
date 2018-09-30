import numpy as np
import tensorflow as tf

import common
from networks import blocks, util
from tensorflow.contrib import layers
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Policy:
    def __init__(self, obs_spec, config, scope):
        self.config = config
        self.scope = scope
        self.s_size = obs_spec["screen"][1]
        self.m_size = obs_spec["minimap"][1]
        self.pi = self.vf = None
        self.epsilon = self.config.epsilon_start
        self.beta = self.config.beta_start
        self.eta = self.config.eta

    def build_placeholders(self):
        self._build_placeholders()

    def build_model(self):
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self.scope):
            self._build_network()

            if not tf.get_variable_scope().name.startswith("global"):
                self._build_loss()

            self.var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def _build_placeholders(self):
        self.obs = common.get_feature_placeholders(self.s_size, self.m_size)
        self.actions = {
            "spatial": tf.placeholder(tf.int32, [None], name="spatial_actions"),
            "non_spatial": tf.placeholder(tf.int32, [None], name="non_spatial_actions")
        }
        self.adv = tf.placeholder(tf.float32, [None], name="advantages")
        self.r = tf.placeholder(tf.float32, [None], name="returns")

    def _build_network(self):
        pass

    def _build_loss(self):
        self.log_prob = self._log_prob()
        self.entropy = self._entropy()

        with tf.variable_scope("policy_loss"):
            self.pi_loss = - tf.reduce_sum(self.log_prob * self.adv)
        with tf.variable_scope("value_loss"):
            self.vf_loss = self.eta * tf.reduce_sum(tf.square(self.vf - self.r))
        with tf.variable_scope("loss"):
            self.loss = self.pi_loss + self.vf_loss - self.entropy * self.beta

    def _log_prob(self):
        with tf.variable_scope("log_prob"):
            spatial_probs = tf.reduce_sum(
                tf.gather(self.pi["spatial"], self.actions["spatial"], axis=1)
            )
            spatial_probs_log = tf.log(spatial_probs)

            non_spatial_probs = tf.reduce_sum(
                tf.gather(self.pi["non_spatial"], self.actions["non_spatial"], axis=1)
            )
            non_spatial_probs_log = tf.log(non_spatial_probs)

            return spatial_probs_log * non_spatial_probs_log

    def _entropy(self):
        with tf.variable_scope("entropy"):
            spatial_entropy = tf.reduce_sum(self.pi["spatial"] * tf.log(self.pi["spatial"]), 1)
            non_spatial_entropy = tf.reduce_sum(self.pi["non_spatial"] * tf.log(self.pi["non_spatial"]), 1)

            return tf.reduce_mean(
                [
                    tf.reduce_mean(spatial_entropy),
                    tf.reduce_mean(non_spatial_entropy)
                ]
            )

    def act(self, obs):
        sess = tf.get_default_session()
        feed_dict = common.populate_feature_dict(obs, self.obs, self.s_size)
        pi, vf = sess.run(
            [self.pi, self.vf],
            feed_dict=feed_dict
        )

        valid_actions = obs["available_actions"]
        if self.config.e_greedy:
            if np.random.random() < self.epsilon:
                act_id, target = util.exploit_distribution(pi, valid_actions, self.s_size)
            else:
                act_id, target = util.exploit_max(pi, valid_actions, self.s_size)
        else:
            act_id, target = util.exploit_distribution(pi, valid_actions, self.s_size)

        return act_id, target, vf

    def value(self, obs):
        sess = tf.get_default_session()
        return sess.run(
            self.vf,
            feed_dict=common.populate_feature_dict(obs, self.obs, self.s_size)
        )


class FullyConv(Policy):
    def __init__(self, obs_spec, config):
        super().__init__(obs_spec, config, scope="fully_conv")

    def _build_network(self):
        m_conv = blocks.cnn(self.obs["minimap"], "minimap_feat")
        s_conv = blocks.cnn(self.obs["screen"], "screen_feat")

        state_representation = blocks.concat(m_conv, s_conv, self.obs["non_spatial"], "state_rep")
        fc = blocks.fully_connected(state_representation, 256, "fc")

        self.pi = {
            "spatial": blocks.spatial_action(state_representation, "spatial_act_pi"),
            "non_spatial": blocks.non_spatial_action(fc, "non_spatial_act_pi")
        }

        self.vf = blocks.build_value(fc, "value")


class FullyConvLSTM(Policy):
    def __init__(self, obs_spec, config):
        super().__init__(obs_spec, config, scope="lstm_conv")

    def _build_network(self):
        m_conv = blocks.cnn(self.obs["minimap"], "minimap_feat")
        s_conv = blocks.cnn(self.obs["screen"], "screen_feat")

        state_representation = blocks.concat(m_conv, s_conv, self.obs["non_spatial"], "state_rep")

        state_conv_LSTM = blocks.conv_LSTM(state_representation, "state_rep_Conv_LSTM")

        fc = blocks.fully_connected(state_conv_LSTM, 256, "fc")
        squeezed_state = tf.squeeze(state_conv_LSTM, axis=1)

        self.pi = {
            "spatial": blocks.spatial_action(squeezed_state, "spatial_act_pi"),
            "non_spatial": blocks.non_spatial_action(fc, "non_spatial_act_pi")
        }

        self.vf = blocks.build_value(fc, "value")


class AtariNet(Policy):
    def __init__(self, obs_spec, config):
        super().__init__(obs_spec, config, scope="atari_net")

    def _build_network(self):
        m_conv = blocks.cnn(self.obs["minimap"], "minimap_feat", kernel_size=[8, 4], stride=[4, 2])
        s_conv = blocks.cnn(self.obs["screen"], "screen_feat", kernel_size=[8, 4], stride=[4, 2])
        non_spatial = blocks.non_spatial_feat_atari(self.obs["non_spatial"], "ns_feat")

        state_representation = tf.concat(
            [
                layers.flatten(m_conv),
                layers.flatten(s_conv),
                non_spatial
            ],
            axis=1,
            name="state_rep"
        )

        fc = blocks.fully_connected(state_representation, 256, "fc")

        spatial_action_x = blocks.spatial_action_atari(fc, self.s_size, "spatial_act_x", transpose=True)
        spatial_action_y = blocks.spatial_action_atari(fc, self.s_size, "spatial_act_y")

        spatial_action = layers.flatten(
            tf.multiply(
                spatial_action_x,
                spatial_action_y
            ),
            scope="spatial_act_pi"
        )

        self.pi = {
            "spatial": spatial_action,
            "non_spatial": blocks.non_spatial_action(fc, "non_spatial_act_pi")
        }

        self.vf = blocks.build_value(fc, "value")
