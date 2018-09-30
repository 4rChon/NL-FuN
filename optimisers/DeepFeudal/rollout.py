from collections import namedtuple
import optimisers.util as util
from pysc2.lib import actions
import time
import common
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_deep_feudal_rollout(rollout, manager_gamma, worker_gamma):
    batch_si = rollout.obs
    batch_a = rollout.actions

    manager_rewards_plus_v = rollout.rewards + [rollout.manager_r]
    worker_rewards_plus_v = rollout.rewards + [rollout.worker_r]

    batch_manager_r = util.discount(manager_rewards_plus_v, manager_gamma)[:-1]
    batch_worker_r = util.discount(worker_rewards_plus_v, worker_gamma)[:-1]

    batch_s = rollout.ss
    batch_g = rollout.gs
    batch_g_prev = rollout.g_prev

    batch_idx = rollout.idx
    features = rollout.features

    return Batch(batch_si, batch_a, batch_manager_r, batch_worker_r,
                 rollout.terminal, batch_s, batch_g, batch_g_prev, batch_idx, features)


Batch = namedtuple("Batch", ["obs", "actions", "manager_returns", "worker_returns",
                             "terminal", "s", "g", "g_prev", "idx", "features"])


def rollout_provider(env, policy, num_local_steps, result_tracker=None):
    last_obs = env.reset()[0].observation
    last_sm_c_g, sm_idx, last_m_c_g, m_idx, last_features = policy.get_initial_features()

    length = 0
    rewards = 0

    size = len(last_obs['screen'][1])

    while True:
        terminal_end = False
        deep_rollout = DeepFeudalRollout(1)
        rollout = DeepFeudalRollout(0)

        for i in range(num_local_steps):
            fetched = policy.act(last_obs, last_sm_c_g, sm_idx, last_m_c_g, m_idx, last_features)

            action, super_manager, manager, worker, features = fetched

            sm_g, sm_s, sm_idx, sm_vf = super_manager
            m_g, m_s, m_idx, m_vf, last_sm_c_g = manager
            pi, last_m_c_g = worker

            act_id, act_target = action
            act_args = common.get_action_arguments(act_id, act_target)
            action = actions.FunctionCall(act_id, act_args)

            timestep = env.step([action])[0]
            obs = timestep.observation
            reward = timestep.reward
            terminal = timestep.last()

            target_index = act_target[1] * size + act_target[0]

            screen_obs = common.screen_obs(obs)
            minimap_obs = common.minimap_obs(obs)
            non_spatial_obs = common.non_spatial_obs(obs, size)

            deep_rollout.add(
                screen_obs, minimap_obs, non_spatial_obs,
                act_id, target_index,
                reward, sm_vf,
                sm_g, sm_s, last_sm_c_g,
                sm_idx, terminal, last_features
            )

            rollout.add(
                screen_obs, minimap_obs, non_spatial_obs,
                act_id, target_index,
                reward, m_vf,
                m_g, m_s, last_m_c_g,
                m_idx, terminal, last_features
            )

            length += 1
            rewards += reward

            last_obs = obs
            last_features = features

            if terminal:
                terminal_end = True
                logger.info("Episode finished. Sum of rewards: %f.", rewards)
                if result_tracker:
                    result_tracker.save(time.time(), rewards)
                rewards = 0
                length = 0
                break

        if not terminal_end:
            sm_r, m_r, w_r = policy.value(last_obs, last_sm_c_g, sm_idx, last_m_c_g, m_idx, last_features)

            rollout.worker_r = w_r
            rollout.manager_r = m_r

            deep_rollout.worker_r = m_r
            deep_rollout.manager_r = sm_r

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout, deep_rollout


class DeepFeudalRollout:
    def __init__(self, level):
        self.level = level

        self.obs = {
            "screen": [],
            "minimap": [],
            "non_spatial": []
        }
        self.actions = {
            "non_spatial": [],
            "spatial": []
        }
        self.rewards = []
        self.values = []

        self.ss = []
        self.gs = []
        self.g_prev = []
        self.idx = []
        self.features = []
        self.manager_r = 0.0
        self.worker_r = 0.0
        self.terminal = False

    def add(self,
            screen_obs, minimap_obs, non_spatial_obs,
            act_id, act_target,
            reward, value, g, s, g_prev, idx, terminal, features):
        self.obs["screen"].extend(screen_obs)
        self.obs["minimap"].extend(minimap_obs)
        self.obs["non_spatial"].extend(non_spatial_obs)

        self.actions["non_spatial"] += [act_id]
        self.actions["spatial"] += [act_target]

        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.gs += [g]
        self.g_prev += [g_prev]
        self.idx += [idx]
        self.ss += [s]

    def extend(self, other):
        self.obs["screen"].extend(other.obs["screen"])
        self.obs["minimap"].extend(other.obs["minimap"])
        self.obs["non_spatial"].extend(other.obs["non_spatial"])

        self.actions["spatial"].extend(other.actions["spatial"])
        self.actions["non_spatial"].extend(other.actions["non_spatial"])

        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.gs.extend(other.gs)
        self.idx.extend(other.idx)
        self.g_prev.extend(other.g_prev)
        self.ss.extend(other.ss)

        self.manager_r = other.manager_r
        self.worker_r = other.worker_r
        self.terminal = other.terminal
        self.features.extend(other.features)
