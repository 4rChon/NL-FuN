from common import util
from common import preprocessing as pp

populate_feature_dict = util.populate_feature_dict
get_action_arguments = util.get_action_arguments
minimap_obs = util.minimap_obs
screen_obs = util.screen_obs
non_spatial_obs = util.non_spatial_obs

get_feature_placeholders = pp.get_feature_placeholders
preprocess_spatial_features = pp.preprocess_spatial_features
preprocess_non_spatial_features = pp.preprocess_non_spatial_features

FORCED_SCALARS = pp.FORCED_SCALARS
INCLUDED_FEATURES = pp.INCLUDED_FEATURES

NON_SPATIAL_FEATURE_SIZE = pp.NON_SPATIAL_FEATURE_SIZE
SCREEN_FEATURE_SIZE = pp.SCREEN_FEATURE_SIZE
MINIMAP_FEATURE_SIZE = pp.MINIMAP_FEATURE_SIZE
