from collections import namedtuple
import numpy as np
from pysc2.lib import actions
import time
import logging

import common
import optimisers.util as util
from optimisers.rollout import PartialRollout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

screen = common.SCREEN_KEY
minimap = common.MINIMAP_KEY


def process_a3c_rollout(rollout, gamma):
    batch_obs = rollout.obs
    batch_a = rollout.actions
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = util.discount(rewards_plus_v, gamma)[:-1]

    vpred_t = np.asarray(rollout.values + [rollout.r])
    rewards = np.asarray(rollout.rewards)
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = util.discount(delta_t, gamma)

    return Batch(batch_obs, batch_a, batch_adv, batch_r, rollout.terminal)


Batch = namedtuple("Batch", ["obs", "actions", "adv", "r", "terminal"])


def rollout_provider(env, policy, num_local_steps, result_tracker=None):
    last_obs = env.reset()[0].observation
    length = 0
    rewards = 0
    size = len(last_obs[screen][1])
    current_sess_episode = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for i in range(num_local_steps):
            act_id, act_target, value_ = policy.act(last_obs)
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

            rollout.add(screen_obs, minimap_obs, non_spatial_obs, act_id, target_index, reward, value_[-1][-1], terminal)
            length += 1
            rewards += reward

            last_obs = obs

            if terminal:
                terminal_end = True
                current_sess_episode += 1
                logger.info("Episode {} finished. Sum of rewards: {}. Length: {}".format(current_sess_episode, rewards, length))
                if result_tracker:
                    result_tracker.save(time.time(), rewards)

                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_obs)[-1]

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout