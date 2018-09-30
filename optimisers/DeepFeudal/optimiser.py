import tensorflow as tf
import logging
import queue
import optimisers.util as util
from optimisers.DeepFeudal.rollout import process_deep_feudal_rollout, rollout_provider
from optimisers.optimiser import Optimiser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepFeudal(Optimiser):
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
            scope="Deep_FeUdal_Optimiser"
        )

        self.prev_deep_batch = None
        self.prev_batch = None

    def _build_summaries(self):
        w_summaries = [
            tf.summary.scalar("Worker_Loss", self.network.worker.loss),
            tf.summary.scalar("Worker_Pi_Loss", self.network.worker.pi_loss),
            tf.summary.scalar("Worker_Value_Loss", self.network.worker.vf_loss),
            tf.summary.scalar("Entropy", self.network.worker.entropy),
            tf.summary.scalar("Learning_Rate", self.learning_rate),
            tf.summary.histogram("Worker Intrinsic Reward", self.network.worker.ri),
            tf.summary.histogram("Worker Reward", self.network.worker.r),
            tf.summary.histogram("Worker Value", self.network.worker.vf),
            tf.summary.histogram("Spatial_Policy", self.network.pi["spatial"]),
            tf.summary.histogram("Non_Spatial_Policy", self.network.pi["non_spatial"]),
            tf.summary.image("Spatial_Policy", tf.reshape(self.network.pi["spatial"], [-1, 32, 32, 1]), max_outputs=1),
            tf.summary.image("Unit Type", tf.reshape(self.network.obs["screen"][2], [-1, 32, 32, 1]), max_outputs=1)
        ]

        if self.config.e_greedy:
            w_summaries += [tf.summary.scalar("Epsilon", self.epsilon)]

        self.w_summary_op = tf.summary.merge(w_summaries)

        m_summaries = [
            tf.summary.scalar("Manager_Loss", self.network.manager.loss),
            tf.summary.scalar("Manager_Pi_Loss", self.network.manager.pi_loss),
            tf.summary.scalar("Manager_Value_Loss", self.network.manager.vf_loss),
            tf.summary.histogram("Manager Intrinsic Reward", self.network.manager.ri),
            tf.summary.histogram("Manager Reward", self.network.manager.r),
            tf.summary.histogram("Manager Value", self.network.manager.vf)
        ]

        self.m_summary_op = tf.summary.merge(m_summaries)

        sm_summaries = [
            tf.summary.scalar("Super_Manager_Loss", self.network.super_manager.loss),
            tf.summary.scalar("Super_Manager_Pi_Loss", self.network.super_manager.pi_loss),
            tf.summary.scalar("Super_Manager_Value_Loss", self.network.super_manager.vf_loss),
            tf.summary.histogram("Super Manager Reward", self.network.super_manager.r),
            tf.summary.histogram("Super Manager Value", self.network.super_manager.vf),
        ]

        self.sm_summary_op = tf.summary.merge(sm_summaries)

    def _build_train_ops(self):
        with tf.variable_scope("train_op"):
            self.sm_train_op = self._build_train_op("super_manager", self.learning_rate,
                                                    self.network.super_manager.loss)
            self.m_train_op = self._build_train_op("manager", self.learning_rate,
                                                   self.network.manager.loss)
            self.w_train_op = self._build_train_op("worker", self.learning_rate,
                                                   self.network.worker.loss, global_step=self.global_step)

    def _build_train_op(self, scope, lr, loss, decay=.9, global_step=None):
        with tf.variable_scope("train_op_{}".format(scope)):
            global_vars = [v for v in self.global_network.var_list if scope in v.name]
            local_vars = [v for v in self.network.var_list if scope in v.name]
            grads = tf.gradients(loss, local_vars)

            # def clip_if_not_none(grad):
            #     if grad is None:
            #         return grad
            #     return tf.clip_by_value(grad, -1.0, 1.0)
            grads, _ = tf.clip_by_global_norm(grads, 1)  # [clip_if_not_none(grad) for grad in grads]
            grads_and_vars = list(zip(grads, global_vars))
            opt = tf.train.RMSPropOptimizer(lr, decay, epsilon=1e-8, centered=True)
            return opt.apply_gradients(grads_and_vars, global_step)

    def pull_batch_from_queue(self):
        rollouts = self.runner.queue.get(timeout=600.0)
        while not rollouts[1].terminal:
            try:
                next_rollouts = self.runner.queue.get_nowait()
                for i in range(len(rollouts)):
                    rollouts[i].extend(next_rollouts[i])
            except queue.Empty:
                break
        return rollouts

    def process(self, sess):
        rollouts = self.pull_batch_from_queue()

        super_manager_gamma = self.config.sm_discount
        manager_gamma = self.config.m_discount
        worker_gamma = self.config.w_discount

        fresh_batch = []
        last_terminal = []
        for r in rollouts:
            if r.level == 0:
                batch_rollout = process_deep_feudal_rollout(
                    r,
                    manager_gamma=manager_gamma,
                    worker_gamma=worker_gamma
                )
                last_terminal += [batch_rollout.terminal]
                fresh_batch += [self.network.update_batch(batch_rollout)]
            elif r.level == 1:
                batch_rollout = process_deep_feudal_rollout(
                    r,
                    manager_gamma=super_manager_gamma,
                    worker_gamma=manager_gamma
                )
                last_terminal += [batch_rollout.terminal]
                fresh_batch += [self.network.update_deep_batch(batch_rollout)]

        # Batch and Deep_Batch need to be reshaped in order to be passed through network.
        # truncate long batches
        def truncate(_batch, _deep_batch):
            shortest_batch = _batch if len(_batch.s_diff) < len(_deep_batch.s_diff) else _deep_batch
            o = shortest_batch.obs
            a = shortest_batch.actions
            b, db = util.truncate_batches(_batch, _deep_batch)
            return o, a, b, db

        # pad short batches
        def pad(_batch, _deep_batch):
            longest_batch = _batch if len(_batch.s_diff) > len(_deep_batch.s_diff) else _deep_batch
            o = longest_batch.obs
            a = longest_batch.actions

            prev_batch = self.prev_batch if not last_terminal[0] else None
            prev_deep_batch = self.prev_deep_batch if not last_terminal[1] else None
            self.prev_deep_batch = _deep_batch
            self.prev_batch = _batch
            b, db = util.pad_batches(_batch, _deep_batch, prev_batch, prev_deep_batch)
            return o, a, b, db

        obs, actions, batch, deep_batch = pad(*fresh_batch)

        w_fetches = [self.w_train_op]
        m_fetches = [self.m_train_op]
        sm_fetches = [self.sm_train_op]

        if self.config.e_greedy:
            w_fetches += [self.epsilon]

        if self.summary_writer:
            w_fetches += [self.w_summary_op, self.global_step]
            sm_fetches += [self.sm_summary_op]
            m_fetches += [self.m_summary_op]

        sm_feed_dict = {
            self.network.obs["screen"]: obs["screen"],
            self.network.obs["minimap"]: obs["minimap"],
            self.network.obs["non_spatial"]: obs["non_spatial"],
            self.network.super_manager.r: deep_batch.manager_returns,
            self.network.super_manager.s_diff: deep_batch.s_diff,
            self.network.super_manager.prev_g: deep_batch.g_prev,
            self.network.super_manager.dilated_idx_in: deep_batch.idx,
            self.network.super_manager.g_in: deep_batch.g_in,
        }

        for i in range(2):
            sm_feed_dict.update({
                self.network.state_in[i]: deep_batch.features[i],
            })

        sm_fetched = sess.run(sm_fetches, feed_dict=sm_feed_dict)

        m_feed_dict = {
            self.network.obs["screen"]: obs["screen"],
            self.network.obs["minimap"]: obs["minimap"],
            self.network.obs["non_spatial"]: obs["non_spatial"],
            self.network.super_manager.prev_g: deep_batch.g_prev,
            self.network.super_manager.g_in: deep_batch.g_in,
            self.network.manager.r: deep_batch.worker_returns,
            self.network.manager.s_diff: batch.s_diff,
            self.network.manager.dilated_idx_in: batch.idx,
            self.network.manager.ri: deep_batch.ri,
        }

        for i in range(2, 4):
            m_feed_dict.update({
                self.network.state_in[i]: deep_batch.features[i],
            })

        m_fetched = sess.run(m_fetches, feed_dict=m_feed_dict)

        batch = fresh_batch[0]
        obs = batch.obs
        actions = batch.actions
        w_feed_dict = {
            self.network.obs["screen"]: obs["screen"],
            self.network.obs["minimap"]: obs["minimap"],
            self.network.obs["non_spatial"]: obs["non_spatial"],
            self.network.actions["spatial"]: actions["spatial"],
            self.network.actions["non_spatial"]: actions["non_spatial"],
            self.network.manager.g_in: batch.g_in,
            self.network.manager.prev_g: batch.g_prev,
            self.network.worker.r: batch.worker_returns,
            self.network.worker.ri: batch.ri,
        }

        for i in range(4, 6):
            w_feed_dict.update({
                self.network.state_in[i]: batch.features[i],
            })

        fetched = sess.run(w_fetches, feed_dict=w_feed_dict)

        if self.config.e_greedy:
            self.network.epsilon = fetched[1]

        if self.summary_writer:
            self.summary_writer.add_summary(tf.Summary.FromString(sm_fetched[-1]), fetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(m_fetched[1]), fetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
            self.summary_writer.flush()

        tf.logging.info("Syncing...")
        sess.run(self.sync)
