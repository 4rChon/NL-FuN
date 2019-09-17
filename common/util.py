import numpy as np

from pysc2.lib import features, actions

from common import preprocessing as pp

SCREEN_KEY = "feature_screen"  # "rgb_screen"
MINIMAP_KEY = "feature_minimap"  # "rgb_minimap"


def minimap_obs(obs):
    """
    Args:
        obs: Observation from the SC2 environment
    Returns:
        Given no forced scalars or excluded features,
        minimap feature layers of shape (1, 33, minimap_size, minimap_size)
    """
    return np.expand_dims(
        pp.preprocess_spatial_features(
            obs[MINIMAP_KEY],
            features.MINIMAP_FEATURES,
            pp.FORCED_SCALARS["minimap"],
            pp.INCLUDED_FEATURES["minimap"]
        ),
        axis=0
    )


def screen_obs(obs):
    """
    Args:
        obs: Observation from the SC2 environment
    Returns:
        Given no forced scalars or excluded features,
        screen feature layers of shape (1, 1907, screen_size, screen_size)
    """
    # shape = (17, size_s, size_s)
    return np.expand_dims(
        pp.preprocess_spatial_features(
            obs[SCREEN_KEY],
            features.SCREEN_FEATURES,
            pp.FORCED_SCALARS["screen"],
            pp.INCLUDED_FEATURES["screen"]
        ),
        axis=0
    )


def non_spatial_obs(obs, size):
    non_spatial = pp.preprocess_non_spatial_features(obs)

    # repeat vector stats over (-1, 1, size, size)
    min_shape = non_spatial.shape[0]
    max_shape = size**2
    last_index = max_shape - max_shape % min_shape
    repeats = max_shape//min_shape

    out_non_spatial = np.zeros(max_shape)
    out_non_spatial[:last_index] = np.concatenate(np.repeat([non_spatial], repeats, axis=0))
    return np.reshape(out_non_spatial, [-1, 1, size, size])


def populate_feature_dict(obs, feats, size):
    return {
        feats["minimap"]: minimap_obs(obs),
        feats["screen"]: screen_obs(obs),
        feats["non_spatial"]: non_spatial_obs(obs, size)
    }


def get_action_arguments(act_id, target):
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append([target[1], target[0]])
            pass
        else:
            act_args.append([0])

    return act_args
