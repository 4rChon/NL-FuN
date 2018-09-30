import tensorflow as tf
import queue
import logging

import networks as nets
import optimisers.util as util
from optimisers.RunnerThread import RunnerThread

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Optimiser:
    def __init__(
            self,
            env,
            config,
            task,
            network,
            rollout_provider,
            summary_writer=None,
            result_tracker=None,
            scope="optimiser"):
        self.config = config
        self.task = task
        self.summary_writer = summary_writer
        self.NetworkClass = getattr(nets, network)
        self.cpu_ix = util.get_device_ix(task, "cpu", cpus=8)
        self.gpu_ix = util.get_device_ix(task, "gpu", gpus=2)

        with tf.variable_scope("global", reuse=tf.AUTO_REUSE):
            self.global_network = util.get_network(self.NetworkClass, env, config)
            self.global_network.build_placeholders()
            self.global_network.build_model()

        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope("local/" + scope + "_" + str(task), reuse=tf.AUTO_REUSE):
            self.network = util.get_network(self.NetworkClass, env, config)
            self.network.build_placeholders()
            self.network.build_model()

            self.runner = RunnerThread(env, self.network, self.config.max_t, rollout_provider, result_tracker)

            self._build_learning_rate()

            if config.e_greedy:
                self._build_e_greedy()
            self._build_sync_op()
            self._build_train_ops()

        if self.summary_writer:
            self._build_summaries()

    def _build_summaries(self):
        pass

    def _build_learning_rate(self):
        if self.config.learning_rate_decay:
            self.learning_rate = tf.train.polynomial_decay(
                self.config.learning_rate, self.global_step,
                end_learning_rate=self.config.learning_rate * 0.5,
                decay_steps=self.config.max_global_eps*self.config.steps_in_ep * 0.5,
                power=1
            )
        else:
            self.learning_rate = self.config.learning_rate

    def _build_e_greedy(self):
        if self.config.epsilon_decay:
            self.epsilon = tf.train.polynomial_decay(
                self.config.epsilon_start, self.global_step,
                end_learning_rate=self.config.epsilon_end,
                decay_steps=self.config.max_global_eps*self.config.steps_in_ep,
                power=1
            )
        else:
            self.epsilon = self.config.epsilon_start

    def _build_sync_op(self):
        # self.sync = tf.group(
        #     *[tf.Print(v1.assign(v2), [v1.name, v2.name, v2], "Syncing: ")
        #       for v1, v2 in zip(self.network.var_list, self.global_network.var_list)]
        # )
        self.sync = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(self.network.var_list, self.global_network.var_list)]
        )

    def _build_train_ops(self):
        pass

    def start(self, sess):
        self.runner.start_runner(sess)

    def pull_batch_from_queue(self):
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        pass
