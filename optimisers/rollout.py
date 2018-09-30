class PartialRollout:
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
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
        self.r = 0.0

        self.terminal = False

    def add(self, screen_obs, minimap_obs, non_spatial_obs, act_id, act_target, reward, value, terminal):
        self.obs["screen"].extend(screen_obs)
        self.obs["minimap"].extend(minimap_obs)
        self.obs["non_spatial"].extend(non_spatial_obs)

        self.actions["non_spatial"] += [act_id]
        self.actions["spatial"] += [act_target]

        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal

    def extend(self, other):
        assert not self.terminal

        self.obs["screen"].extend(other.obs["screen"])
        self.obs["minimap"].extend(other.obs["minimap"])
        self.obs["non_spatial"].extend(other.obs["non_spatial"])

        self.actions["spatial"].extend(other.actions["spatial"])
        self.actions["non_spatial"].extend(other.actions["non_spatial"])

        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal

