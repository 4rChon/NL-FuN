import tensorflow as tf
import logging
from optimisers.A3C.rollout import process_a3c_rollout, rollout_provider
from optimisers.optimiser import Optimiser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class A3C(Optimiser):
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
            scope="A3C_Optimiser"
        )

    def _build_summaries(self):
        self.summaries = [
            tf.summary.scalar("Policy_Loss", self.network.pi_loss),
            tf.summary.scalar("Value_Loss", self.network.vf_loss),
            tf.summary.scalar("Entropy", self.network.entropy),
            tf.summary.scalar("Learning_Rate", self.learning_rate),
            tf.summary.histogram("Value", self.network.vf),
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
            grads = tf.gradients(self.network.loss, self.network.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_and_vars = list(zip(grads, self.global_network.var_list))
            opt = tf.train.RMSPropOptimizer(self.learning_rate, 1.0)
            self.train_op = opt.apply_gradients(grads_and_vars, self.global_step)

    def process(self, sess):
        rollout = self.pull_batch_from_queue()
        batch = process_a3c_rollout(rollout, self.config.gamma)

        fetches = [self.train_op]
        if self.config.e_greedy:
            fetches += [self.epsilon]
        if self.summary_writer:
            fetches += [self.summary_op, self.global_step]

        feed_dict = {
            self.network.obs["minimap"]: batch.obs["minimap"],
            self.network.obs["screen"]: batch.obs["screen"],
            self.network.obs["non_spatial"]: batch.obs["non_spatial"],
            self.network.actions["spatial"]: batch.actions["spatial"],
            self.network.actions["non_spatial"]: batch.actions["non_spatial"],
            self.network.adv: batch.adv,
            self.network.r: batch.r
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if self.config.e_greedy:
            self.network.epsilon = fetched[1]

        if self.summary_writer:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
            self.summary_writer.flush()

        tf.logging.info("Syncing...")
        sess.run(self.sync)
