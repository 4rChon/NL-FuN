import tensorflow as tf
from networks import blocks
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def dcos_fn(g, s_diff, nan_epsilon):
    # manager policy loss
    dot = tf.reduce_sum(tf.multiply(s_diff, g), axis=1)
    # add epsilon in before the norm, otherwise can get nans when
    # g is all zeros and try to backprop the norm
    # can also prevent by stopping gradient, but this actually
    # should be included as part of the backprop
    gcut = g + nan_epsilon
    # also have to epsilon here as well in case s_diff is 0
    mag = tf.norm(s_diff, axis=1) * tf.norm(gcut, axis=1) + nan_epsilon
    return dot / mag


class Citizen:
    def __init__(self, config, scope):
        logger.info("Building %s..." % scope)

        self.scope = scope
        self.use_batch_norm = config.use_batch_norm
        self.dropout_keep_prob = config.dropout_keep_prob
        self.decay_steps = config.decay_steps
        self.nan_epsilon = config.nan_epsilon

        # hyperparams
        self.eta = None
        self.lstm_size = None
        self.s = None
        self.s_dim = None

        # tensor ops
        self.r = None
        self.s = None
        self.vf = None
        self.vf_loss = None
        self.pi_loss = None

    def build_value(self, x):
        logger.info("\tBuilding %s value..." % self.scope)
        self.vf = blocks.build_value(x, "VF_%s" % self.scope)


class SuperManager(Citizen):
    def __init__(self, config, scope="super_manager"):
        super().__init__(config, scope)
        # hyperparams
        self.g_dim = config.sm_goal_dim
        self.eta = np.float32(config.sm_eta)
        self.lstm_size = config.sm_lstm_size
        self.s_dim = config.sm_percept_dim
        self.c = config.sm_horizon

        self.g_eps_decay = config.sm_g_eps_decay
        self.g_eps_start = config.sm_g_eps_start
        self.g_eps_end = config.sm_g_eps_end

        self.random_goals = config.sm_random_goals

        # placeholders
        self.prev_g = None
        self.g_in = None
        self.s_diff = None
        # # lstm placeholders
        self.dilated_idx_in = None
        self.state_in = None

        # tensor ops
        self.state_init = None
        self.state_out = None
        self.dilated_idx_out = None
        self.g = None
        self.last_c_g = None
        self.g_sum = None

    def build_placeholders(self):
        logger.info("Building %s placeholders..." % self.scope)
        self.r = tf.placeholder(tf.float32, (None,), "%s_returns" % self.scope)
        self.prev_g = tf.placeholder(tf.float32, (None, self.c-1, self.g_dim), "prev_g")  # previous goal
        self.g_in = tf.placeholder(tf.float32, (None, self.g_dim), "g_in")

        if not tf.get_variable_scope().name.startswith("global"):
            # difference between last two goals
            self.s_diff = tf.placeholder(tf.float32, (None, self.g_dim), "%s_state_diffs" % self.scope)

        self.state_in = [
            tf.placeholder(tf.float32, (self.c, self.lstm_size), "%s_in_c" % self.scope),
            tf.placeholder(tf.float32, (self.c, self.lstm_size), "%s_in_h" % self.scope)
        ]
        self.dilated_idx_in = tf.placeholder(tf.int32, (), '%s_didx' % self.scope)

    def build_perception(self, sub_citizen_state):
        logger.info("Building %s perception..." % self.scope)
        with tf.variable_scope(self.scope):
            self.s = blocks.manager_percept(sub_citizen_state, self.s_dim, "%s_percept" % self.scope)

    def build_model(self, state_in, state_out):
        logger.info("Building %s model..." % self.scope)
        with tf.variable_scope(self.scope):
            self.build_value(self.s)

            # Calculate manager output g
            x = tf.expand_dims(self.s, [0])

            g_hat, self.state_init, self.state_out, self.dilated_idx_out = blocks.DilatedLSTM(
                x,
                self.lstm_size,
                self.state_in,
                self.dilated_idx_in,
                chunks=self.c,
                norm=self.use_batch_norm,
                dropout=self.dropout_keep_prob
            )

            # g^_t, (3) from 1703.01161
            # g_t, (3) from 1703.01161
            if self.use_batch_norm:
                g_hat = tf.contrib.layers.batch_norm(g_hat, scope="super_manager_batch_norm")

            if self.random_goals:
                if self.g_eps_decay:
                    g_eps = tf.train.polynomial_decay(
                        self.g_eps_start,
                        tf.train.get_or_create_global_step(),
                        end_learning_rate=self.g_eps_end,
                        decay_steps=self.decay_steps,
                        power=1
                    )
                else:
                    g_eps = self.g_eps_start

                g_hat = tf.cond(tf.random_uniform(()) < g_eps,
                                lambda: tf.random_normal(tf.shape(g_hat)),
                                lambda: g_hat)

            self.g = g_hat/tf.stop_gradient(tf.norm(g_hat, axis=1, keep_dims=True))

            state_in.extend([
                self.state_in[0],
                self.state_in[1]
            ])

            state_out.extend([
                self.state_out[0],
                self.state_out[1]
            ])

    def get_w(self, k, scope):
        cut_g = tf.expand_dims(self.g_in, [1])
        gstack = tf.concat(
            [
                tf.zeros((tf.shape(cut_g)[0], self.c, self.g_dim)),
                self.prev_g,
                cut_g
            ],
            axis=1
        )

        self.last_c_g = gstack[:, -(self.c - 1):]
        self.g_sum = tf.reduce_sum(gstack, axis=1)

        phi = tf.get_variable("%s_phi" % scope, (self.g_dim, k),
                              initializer=blocks.normalized_columns_initializer())

        # Calculate w
        return tf.matmul(self.g_sum, phi)

    def build_loss(self):
        logger.info("Building %s loss..." % self.scope)
        dcos = dcos_fn(self.g, self.s_diff, self.nan_epsilon)

        cutoff_vf = tf.reshape(tf.stop_gradient(self.vf), [-1])
        self.pi_loss = -tf.reduce_sum((self.r - cutoff_vf) * dcos)

        # manager value loss
        A = self.r - self.vf
        self.vf_loss = self.eta * tf.reduce_sum(tf.square(A))


class Manager(Citizen):
    def __init__(self, config, scope='manager'):
        super().__init__(config, scope)

        # hyperparams
        self.k = config.k
        self.eta = np.float32(config.m_eta)
        self.lstm_size = config.m_lstm_size
        self.s_dim = config.m_percept_dim

        # related to manager
        self.random_goals = config.m_random_goals
        self.g_dim = config.m_goal_dim
        self.g_eps_decay = config.m_g_eps_decay
        self.g_eps_start = config.m_g_eps_start
        self.g_eps_end = config.m_g_eps_end
        self.c = config.m_horizon

        # related to worker
        self.alpha_decay = config.m_alpha_decay
        self.alpha_start = config.m_alpha_decay

        # placeholders
        # related to manager
        self.prev_g = None
        self.g_in = None
        self.s_diff = None
        # # lstm placeholders
        self.dilated_idx_in = None
        self.state_in = None

        # related to worker
        self.ri = None

        # tensor ops
        self.state_init = None
        self.state_out = None
        self.dilated_idx_out = None
        self.g = None
        self.last_c_g = None
        self.g_sum = None
        self.U = None

    def build_placeholders(self):
        logger.info("Building %s placeholders..." % self.scope)
        self.r = tf.placeholder(tf.float32, (None,), "%s_returns" % self.scope)
        self.prev_g = tf.placeholder(tf.float32, (None, self.c-1, self.g_dim), "%s_prev_g" % self.scope)  # previous goal
        self.g_in = tf.placeholder(tf.float32, (None, self.g_dim), "%s_g_in" % self.scope)

        if not tf.get_variable_scope().name.startswith("global"):
            # difference between last two goals
            self.s_diff = tf.placeholder(tf.float32, (None, self.g_dim), "%s_s_diffs" % self.scope)

        self.state_in = [
            tf.placeholder(tf.float32, (self.c, self.lstm_size), "%s_in_c" % self.scope),
            tf.placeholder(tf.float32, (self.c, self.lstm_size), "%s_in_h" % self.scope)
        ]
        self.dilated_idx_in = tf.placeholder(tf.int32, (), '%s_didx' % self.scope)

        self.ri = tf.placeholder(tf.float32, (None,), name="%s_ri" % self.scope)  # intrinsic reward

    def build_perception(self, sub_citizen_state):
        logger.info("Building %s perception..." % self.scope)
        with tf.variable_scope(self.scope):
            self.s = blocks.manager_percept(sub_citizen_state, self.s_dim, "%s_percept" % self.scope)

    def build_model(self, w_func, state_in, state_out):
        logger.info("Building %s model..." % self.scope)
        with tf.variable_scope(self.scope):
            self.build_value(self.s)

            x = tf.expand_dims(self.s, [0])

            self.U, self.state_init, self.state_out, self.dilated_idx_out = blocks.DilatedLSTM(
                x,
                self.lstm_size,
                self.state_in,
                self.dilated_idx_in,
                chunks=self.c,
                norm=self.use_batch_norm,
                dropout=self.dropout_keep_prob,
            )

            # g^_t, (3) from 1703.01161
            # g_t, (3) from 1703.01161
            self.U = tf.layers.dense(
                inputs=self.U,
                units=self.g_dim * self.k,
                activation=tf.nn.relu,
                name='%s_hidden_g_hat' % self.scope
            )

            if self.use_batch_norm:
                self.U = tf.contrib.layers.batch_norm(self.U, scope="manager_batch_norm")

            w = w_func(self.k, self.scope)
            w = tf.expand_dims(w, 2)

            self.U = tf.reshape(self.U, [-1, self.g_dim, self.k], name='U_%s' % self.scope)

            g_hat = tf.reshape(tf.matmul(self.U, w), [-1, self.g_dim])

            if self.random_goals:
                if self.g_eps_decay:
                    g_eps = tf.train.polynomial_decay(
                        self.g_eps_start,
                        tf.train.get_or_create_global_step(),
                        end_learning_rate=self.g_eps_end,
                        decay_steps=self.decay_steps,
                        power=1
                    )
                else:
                    g_eps = self.g_eps_start

                g_hat = tf.cond(tf.random_uniform(()) < g_eps,
                                lambda: tf.random_normal(tf.shape(g_hat)),
                                lambda: g_hat)

            self.g = g_hat/tf.stop_gradient(tf.norm(g_hat, axis=1, keep_dims=True))

            state_in.extend([
                self.state_in[0],
                self.state_in[1]
            ])

            state_out.extend([
                self.state_out[0],
                self.state_out[1]
            ])

    def build_loss(self):
        logger.info("Building %s loss..." % self.scope)
        dcos = dcos_fn(self.g, self.s_diff, self.nan_epsilon)

        cutoff_vf = tf.reshape(tf.stop_gradient(self.vf), [-1], name="%s_cutoff_vf" % self.scope)

        if self.alpha_decay:
            alpha = tf.train.polynomial_decay(
                self.alpha_start, tf.train.get_or_create_global_step(),
                end_learning_rate=self.alpha_start*0.5,
                decay_steps=self.decay_steps,
                power=1
            )
        else:
            alpha = self.alpha_start

        ri_weighted = alpha * self.ri
        self.pi_loss = (self.r + ri_weighted - cutoff_vf) * dcos
        self.pi_loss = -tf.reduce_sum(self.pi_loss, name="%s_loss" % self.scope)

        # manager value loss
        A = (self.r + ri_weighted) - self.vf
        self.vf_loss = self.eta * tf.reduce_sum(tf.square(A))

    def get_w(self, k, scope):
        cut_g = tf.expand_dims(self.g_in, [1])
        gstack = tf.concat(
            [
                tf.zeros((tf.shape(cut_g)[0], self.c, self.g_dim)),
                self.prev_g,
                cut_g
            ],
            axis=1
        )

        self.last_c_g = gstack[:, -(self.c - 1):]
        self.g_sum = tf.reduce_sum(gstack, axis=1)

        phi = tf.get_variable("%s_phi" % scope, (self.g_dim, k),
                              initializer=blocks.normalized_columns_initializer())

        # Calculate w
        return tf.matmul(self.g_sum, phi)


class Worker(Citizen):
    def __init__(self, config, num_actions, s_size, m_size, scope="worker"):
        super().__init__(config, scope)
        # hyperparams
        self.k = config.k
        self.num_actions = num_actions
        self.s_size = s_size
        self.m_size = m_size

        self.eta = np.float32(config.eta)
        self.lstm_size = config.w_lstm_size
        self.s_dim = config.z_dim

        self.alpha_decay = config.w_alpha_decay
        self.alpha_start = config.w_alpha_start

        self.beta_decay = config.w_beta_decay
        self.beta_start = config.w_beta_start

        # placeholders
        self.ri = None

        # tensor ops
        self.s_fc = None
        self.U_s = None
        self.U_ns = None
        self.pi = None
        self.log_pi = None
        self.entropy = None

        # lstm
        self.lstm = None

    def build_placeholders(self):
        logger.info("Building %s placeholders..." % self.scope)
        self.r = tf.placeholder(tf.float32, (None,), "%s_returns" % self.scope)
        self.ri = tf.placeholder(tf.float32, (None,), name="%s_ri" % self.scope)  # intrinsic reward

    def build_perception(self, obs):
        logger.info("Building %s perception..." % self.scope)
        with tf.variable_scope(self.scope):
            m_conv = blocks.cnn(obs["minimap"], "minimap_feat")
            s_conv = blocks.cnn(obs["screen"], "screen_feat")

            self.s = blocks.concat(m_conv, s_conv, obs["non_spatial"], "percept_concat")
            if self.use_batch_norm:
                self.s = tf.contrib.layers.batch_norm(self.s)
            self.s_fc = blocks.fully_connected(self.s, self.s_dim, "%s_perception" % self.scope)
            if self.use_batch_norm:
                self.s_fc = tf.contrib.layers.batch_norm(self.s_fc)

    def build_model(self, w_func, obs, state_in, state_out):
        logger.info("Building %s model..." % self.scope)
        with tf.variable_scope(self.scope):
            # Calculate U
            self.lstm = blocks.SingleStepConvLSTM(
                self.s,
                size=self.s_size,
                step_size=tf.shape(obs["minimap"])[:1],
                filters=1,
                scope="%s_lstm" % self.scope
            )

            if self.use_batch_norm:
                self.lstm.output = tf.contrib.layers.batch_norm(self.lstm.output, scope='%s_lstm_batch_norm' % self.scope)

            fc = blocks.fully_connected(self.lstm.output, self.s_dim, "fc")

            self.U_s = tf.layers.conv2d(
                inputs=self.lstm.output,
                filters=self.k,
                kernel_size=1,
                padding='SAME',
                name="spatial_flat_logits_hidden"
            )

            U_fc = blocks.fully_connected(self.lstm.output, self.num_actions * self.k, scope="U_fc")

            if self.use_batch_norm:
                self.U_s = tf.contrib.layers.batch_norm(self.U_s, scope='%s_spatial_batch_norm' % self.scope)
                U_fc = tf.contrib.layers.batch_norm(U_fc, scope='%s_non_spatial_batch_norm' % self.scope)
                fc = tf.contrib.layers.batch_norm(fc, scope='%s_fc_batch_norm' % self.scope)

            self.build_value(fc)

            self.U_s = tf.reshape(self.U_s, [-1, self.s_size ** 2, self.k], name='U_s')
            self.U_ns = tf.reshape(U_fc, [-1, self.num_actions, self.k], name='U_ns')

            w = w_func(self.k, self.scope)
            w = tf.expand_dims(w, 2)

            # calculate policy and sample
            s_logits = tf.reshape(tf.matmul(self.U_s, w), [-1, self.s_size ** 2])
            ns_logits = tf.reshape(tf.matmul(self.U_ns, w), [-1, self.num_actions])

            # Calculate policy
            self.pi = {
                "spatial": tf.nn.softmax(s_logits),
                "non_spatial": tf.nn.softmax(ns_logits)
            }

            self.log_pi = {
                "spatial": tf.nn.log_softmax(s_logits),
                "non_spatial": tf.nn.log_softmax(ns_logits)
            }

            # add worker c, h to state in and out
            state_in.extend([
                self.lstm.state_in[0],
                self.lstm.state_in[1],
            ])

            state_out.extend([
                self.lstm.state_out[0],
                self.lstm.state_out[1],
            ])

    def build_loss(self, actions):
        logger.info("Building %s loss..." % self.scope)
        cutoff_vf = tf.reshape(tf.stop_gradient(self.vf), [-1])
        spatial_probs = tf.gather(
            self.log_pi["spatial"],
            actions["spatial"],
            name="gather_spatial",
            axis=1
        )
        non_spatial_probs = tf.gather(
            self.log_pi["non_spatial"],
            actions["non_spatial"],
            name="gather_non_spatial",
            axis=1
        )
        log_p = tf.reduce_sum(spatial_probs) * tf.reduce_sum(non_spatial_probs)

        if self.alpha_decay:
            alpha = tf.train.polynomial_decay(
                self.alpha_start, tf.train.get_or_create_global_step(),
                end_learning_rate=self.alpha_start*0.5,
                decay_steps=self.decay_steps,
                power=1
            )
        else:
            alpha = self.alpha_start

        ri_weighted = alpha * self.ri
        self.pi_loss = (self.r + ri_weighted - cutoff_vf) * log_p
        self.pi_loss = -tf.reduce_sum(self.pi_loss)

        # worker value loss
        A = (self.r + ri_weighted) - self.vf
        self.vf_loss = self.eta * tf.reduce_sum(tf.square(A))

    def build_entropy(self):
        logger.info("\tBuilding %s entropy..." % self.scope)
        spatial_entropy = tf.reduce_sum(self.pi["spatial"] * self.log_pi["spatial"], 1)
        non_spatial_entropy = tf.reduce_sum(self.pi["non_spatial"] * self.log_pi["non_spatial"], 1)

        if self.beta_decay:
            beta = tf.train.polynomial_decay(
                self.beta_start, tf.train.get_or_create_global_step(),
                end_learning_rate=self.beta_start*0.5,
                decay_steps=self.decay_steps,
                power=1
            )
        else:
            beta = self.beta_start

        self.entropy = (-tf.reduce_mean([spatial_entropy, non_spatial_entropy])) * beta
