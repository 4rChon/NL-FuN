import numpy as np


def exploit_distribution(policy, valid_actions, size):
    # Mask actions
    non_spatial_policy = policy["non_spatial"][-1][valid_actions]

    # Normalize probabilities
    non_spatial_probs = non_spatial_policy/np.sum(non_spatial_policy)

    # Choose from normalized distribution
    act_id = np.random.choice(valid_actions, p=non_spatial_probs)
    target = np.random.choice(np.arange(len(policy["spatial"][-1])), p=policy["spatial"][-1])

    # Resize to provided resolution
    coords = [int(target // size), int(target % size)]

    return act_id, coords


def exploit_max(policy, valid_actions, size):
    # Choose 'best' valid action
    act_id = valid_actions[np.argmax(policy["non_spatial"][-1][valid_actions])]
    target = np.argmax(policy["spatial"][-1])

    # Resize to provided resolution
    # Example:
    #   target = 535 -> 535 // 64 = 8, 535 % 64 = 24
    #   target = [8, 24]
    target = [int(target // size), int(target % size)]

    return act_id, target