import tensorflow as tf
import logging
import numpy as np
from optimisers.Feudal.rollout import process_feudal_rollout, rollout_provider
from optimisers.optimiser import Optimiser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Feudal(Optimiser):
    def __init__(
            self,
            env,
            config,
            task,
            network,
            summary_writer=None,
            result_tracker=None):
        super().__init__(
            env=env,
            config=config,
            task=task,
            network=network,
            rollout_provider=rollout_provider,
            summary_writer=summary_writer,
            result_tracker=result_tracker,
            scope="FeUdal_Optimiser"
        )

    def _build_summaries(self):
        self.summaries = [
            tf.summary.scalar("Manager_Loss", self.network.manager.loss),
            tf.summary.scalar("Manager_Value_Loss", self.network.manager.vf_loss),
            tf.summary.scalar("Worker_Loss", self.network.worker.loss),
            tf.summary.scalar("Worker_Value_Loss", self.network.worker.vf_loss),
            tf.summary.scalar("Entropy", self.network.worker.entropy),
            tf.summary.scalar("Learning_Rate", self.learning_rate),
            tf.summary.histogram("Intrinsic Reward", self.network.worker.ri),
            tf.summary.histogram("Manager Reward", self.network.manager.r),
            tf.summary.histogram("Worker Reward", self.network.worker.r),
            tf.summary.histogram("Manager Value", self.network.manager.vf),
            tf.summary.histogram("Worker Value", self.network.worker.vf),
            tf.summary.histogram("Spatial_Policy", self.network.pi["spatial"]),
            tf.summary.histogram("Non_Spatial_Policy", self.network.pi["non_spatial"]),
            tf.summary.image("Spatial_Policy", tf.reshape(self.network.pi["spatial"], [-1, 32, 32, 1]), max_outputs=1),
            tf.summary.image("Unit Type", tf.reshape(self.network.obs["screen"][2], [-1, 32, 32, 1]), max_outputs=1)
        ]

        if self.config.e_greedy:
            tf.summary.scalar("Epsilon", self.epsilon)

        self.summary_op = tf.summary.merge_all()

    def _build_train_ops(self):
        with tf.variable_scope("train_op"):
            self.m_train_op = self._build_train_op("manager", self.learning_rate, self.network.manager.loss)
            self.w_train_op = self._build_train_op("worker", self.learning_rate, self.network.worker.loss, global_step=self.global_step)

    def _build_train_op(self, scope, lr, loss, decay=1., global_step=None):
        with tf.variable_scope("train_op_{}".format(scope)):
            global_vars = [v for v in self.global_network.var_list if scope in v.name]
            local_vars = [v for v in self.network.var_list if scope in v.name]
            grads = tf.gradients(loss, local_vars)

            def clip_if_not_none(grad):
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -1.0, 1.0)
            grads = [clip_if_not_none(grad) for grad in grads]
            grads_and_vars = list(zip(grads, global_vars))
            opt = tf.train.RMSPropOptimizer(lr, decay)
            return opt.apply_gradients(grads_and_vars, global_step)

    def process(self, sess):
        rollout = self.pull_batch_from_queue()
        batch = process_feudal_rollout(
            rollout,
            manager_gamma=self.config.manager_discount,
            worker_gamma=self.config.worker_discount
        )
        batch = self.network.update_batch(batch)

        fetches = [self.w_train_op]
        if self.config.e_greedy:
            fetches += [self.epsilon]
        if self.summary_writer:
            fetches += [self.summary_op, self.global_step]

        feed_dict = {
            self.network.obs["screen"]: batch.obs["screen"],
            self.network.obs["minimap"]: batch.obs["minimap"],
            self.network.obs["non_spatial"]: batch.obs["non_spatial"],
            self.network.actions["spatial"]: batch.actions["spatial"],
            self.network.actions["non_spatial"]: batch.actions["non_spatial"],
            self.network.manager.r: batch.manager_returns,
            self.network.worker.r: batch.worker_returns,
            self.network.manager.s_diff: batch.s_diff,
            self.network.manager.prev_g: batch.g_prev,
            self.network.worker.ri: batch.ri,
            self.network.manager.dilated_idx_in: batch.idx,
            self.network.manager.g_in: batch.g_in
        }

        for i in range(len(self.network.state_in)):
            feed_dict.update({
                self.network.state_in[i]: batch.features[i],
            })

        sess.run(self.m_train_op, feed_dict=feed_dict)
        fetched = sess.run(fetches, feed_dict=feed_dict)

        if self.config.e_greedy:
            self.network.epsilon = fetched[2]

        if self.summary_writer:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
            self.summary_writer.flush()

        tf.logging.info("Syncing...")
        sess.run(self.sync)
