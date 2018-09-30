import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import numpy as np

from pysc2.lib import actions


def cnn(feature, scope, padding="same", kernel_size=None, stride=None):
    with tf.variable_scope(scope):
        preprocess = tf.layers.conv2d(
            tf.transpose(feature, [0, 2, 3, 1]),
            filters=1,
            kernel_size=1,
            padding="SAME",
            name="preprocess"
        )

        if kernel_size is None:
            kernel_size = [5, 3]
        if stride is None:
            stride = [1, 1]

        conv1 = tf.layers.conv2d(
            preprocess,
            filters=16,
            kernel_size=kernel_size[0],
            strides=stride[0],
            padding=padding,
            name="conv1"
        )

        return tf.layers.conv2d(
            conv1,
            filters=32,
            kernel_size=kernel_size[1],
            strides=stride[1],
            padding=padding,
            name="conv2"
        )


def concat(m_spatial, s_spatial, non_spatial, scope):
        return tf.concat([
            m_spatial,
            s_spatial,
            tf.transpose(non_spatial, [0, 2, 3, 1])
        ], axis=3, name=scope)


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def fully_connected(state_representation, num_outputs, scope):
    return layers.fully_connected(
        layers.flatten(state_representation),
        num_outputs=num_outputs,
        activation_fn=tf.nn.relu,
        scope=scope
    )


def spatial_action_atari(fc, s_size, scope, transpose=False):
    shape = [-1, s_size, 1] if transpose else [-1, 1, s_size]
    return tf.reshape(
        layers.fully_connected(
            fc,
            num_outputs=s_size,
            activation_fn=tf.nn.softmax
        ),
        shape,
        name=scope
    )


def non_spatial_feat_atari(non_spatial, scope):
    return layers.fully_connected(
        layers.flatten(non_spatial),
        num_outputs=len(actions.FUNCTIONS),
        activation_fn=tf.tanh,
        scope=scope
)


def conv_LSTM(state, scope):
    with tf.variable_scope(scope):
        state_shape = state.shape.as_list()

        channels = state_shape[-1]

        conv_LSTM_cell = rnn.ConvLSTMCell(
            conv_ndims=2,
            input_shape=state_shape[1:],    # exclude batch size
            output_channels=channels,
            kernel_shape=[1, 1],
        )

        # State will have the following shape:
        #   [timesteps, batch size, width, height, channels]
        #   timesteps = rollout size
        #   batch size = 1 (online learning)
        #   width = height = size_s = size_m
        #   channels = depends on preprocessing, number of features, etc...
        state = tf.expand_dims(state, 1)
        conv_LSTM_out, _ = tf.nn.dynamic_rnn(
            conv_LSTM_cell,
            state,
            time_major=True,
            dtype=state.dtype
        )

        return conv_LSTM_out


def spatial_action(state_representation, scope):
    with tf.variable_scope(scope):
        return tf.nn.softmax(
            layers.flatten(
                tf.layers.conv2d(
                    state_representation,
                    filters=1,
                    kernel_size=1,
                )
            )
        )


def non_spatial_action(fc, scope):
    return layers.fully_connected(
        fc,
        num_outputs=len(actions.FUNCTIONS),
        activation_fn=tf.nn.softmax,
        scope=scope
    )


def build_value(fc, scope):
    return layers.fully_connected(
        fc,
        num_outputs=1,
        activation_fn=None,
        scope=scope
    )


def manager_percept(z, s_dim, scope):
    return layers.fully_connected(
        z,
        num_outputs=s_dim,
        activation_fn=tf.nn.relu,
        scope=scope
    )


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME",
        dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1],
            int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype,
            tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections)
        # adding initialization to bias because otherwise the network will
        # output all zeros, which is normally fine, but in the feudal case
        # this yields a divide by zero error. Bounds are just small random.
        b = tf.get_variable("b", [1, 1, 1, num_filters],
            initializer=tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


class SingleStepConvLSTM:
    def __init__(self, state, size, step_size, filters, scope):
        state_shape = state.shape.as_list()
        lstm = rnn.ConvLSTMCell(
            conv_ndims=2,
            input_shape=state_shape[1:],  # exclude batch size
            output_channels=filters,
            kernel_shape=[3, 3]
        )

        c_size = [int(c) for c in lstm.state_size.c]
        h_size = [int(h) for h in lstm.state_size.h]

        c_init = np.zeros([1] + c_size, np.float32)
        h_init = np.zeros([1] + h_size, np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(
            tf.float32,
            shape=[1] + c_size,
            name='c_in_{}'.format(scope)
        )

        h_in = tf.placeholder(
            tf.float32,
            shape=[1] + h_size,
            name='h_in_{}'.format(scope)
        )
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        # State will have the following shape:
        #   [timesteps, batch size, width, height, channels]
        #   timesteps = rollout size
        #   batch size = 1 (online learning)
        #   width = height = size_s = size_m
        #   channels = depends on preprocessing, number of features, etc...
        state = tf.expand_dims(state, 1)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm,
            state,
            time_major=True,
            initial_state=state_in,
            sequence_length=step_size
        )
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size, size, filters])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs


class SingleStepLSTM:
    def __init__(self, x, size, step_size, scope=""):
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True, name="rnn_{}".format(scope))

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(
            tf.float32,
            shape=[1, lstm.state_size.c],
            name='c_in_{}'.format(scope)
        )

        h_in = tf.placeholder(
            tf.float32,
            shape=[1, lstm.state_size.h],
            name='h_in_{}'.format(scope)
        )
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm,
            x,
            initial_state=state_in,
            sequence_length=step_size,
            time_major=False
        )

        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs


def DilatedLSTM(s_t, size, state_in, idx_in, chunks=8, norm=False, dropout=1.0):
    # lstm = rnn.LSTMCell(size, state_is_tuple=True)
    lstm = rnn.LayerNormBasicLSTMCell(size, layer_norm=norm, activation=tf.nn.relu, dropout_keep_prob=dropout)
    c_init = np.zeros((chunks, lstm.state_size.c), np.float32)
    h_init = np.zeros((chunks, lstm.state_size.h), np.float32)
    state_init = [c_init, h_init]

    def dlstm_scan_fn(previous_output, current_input):
        i = previous_output[2]
        c = previous_output[1][0]
        h = previous_output[1][1]
        # old_state = [tf.expand_dims(c[i],[0]),tf.expand_dims(h[i],[0])]
        old_state = rnn.LSTMStateTuple(tf.expand_dims(c[i], [0]), tf.expand_dims(h[i], [0]))
        out, state_out = lstm(current_input, old_state)
        c = tf.stop_gradient(c)
        h = tf.stop_gradient(h)
        co = state_out[0]
        ho = state_out[1]
        col = tf.expand_dims(tf.one_hot(i, chunks), [1])
        new_mask = col
        for _ in range(size - 1):
            new_mask = tf.concat([new_mask, col], axis=1)
        old_mask = 1 - new_mask
        c_out = c * old_mask + co * new_mask
        h_out = h * old_mask + ho * new_mask
        state_out = [c_out, h_out]
        i += tf.constant(1)
        new_i = tf.mod(i, chunks)
        out = tf.reduce_mean(h_out, axis=0)
        return out, state_out, new_i

    rnn_outputs, final_states, out_idx = tf.scan(dlstm_scan_fn,
                                                 tf.transpose(s_t, [1, 0, 2]),
                                                 initializer=(state_in[1][0], state_in, idx_in))
    state_out = [final_states[0][0, :, :], final_states[1][0, :, :]]
    return rnn_outputs, state_init, state_out, out_idx[-1]
