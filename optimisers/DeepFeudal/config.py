from optimisers.config import OptConfig, logU
from numpy import random


class DeepConfig(OptConfig):
    def __init__(self, FLAGS, cluster):
        super().__init__(FLAGS, cluster)
        self.max_t = 600
        self.steps_in_ep = 1
        self.decay_steps = self.max_global_eps * self.steps_in_ep

        # Feudal:
        self.k = 16  # Dimensionality of w
        self.z_dim = 256  # Percept dimension
        self.nan_epsilon = 1e-8  # To prevent nans

        # # Super Manager:
        self.sm_lstm_size = 256
        self.sm_discount = 0.999
        self.sm_eta = 0.5  # Super Manager value function weight

        self.sm_percept_dim = 256
        self.sm_goal_dim = 256
        self.sm_horizon = 200

        self.sm_random_goals = True
        self.sm_g_eps_start = 0.005
        self.sm_g_eps_end = 0
        self.sm_g_eps_decay = True

        # # Manager:
        self.m_lstm_size = 256
        self.m_discount = 0.99
        self.m_eta = 0.5  # Manager value function weight

        self.m_percept_dim = 256    # Manager percept dimension
        self.m_goal_dim = 256        # Manager goal dimension
        self.m_horizon = 40         # Manager horizon

        self.m_random_goals = True
        self.m_g_eps_start = 0.001
        self.m_g_eps_end = 0
        self.m_g_eps_decay = True

        self.m_alpha_start = 0.8  # Manager Intrinsic reward weight
        self.m_alpha_decay = False

        self.m_beta_start = 0.001  # Manager entropy decay
        self.m_beta_decay = False

        self.w_alpha_start = 0.8  # Worker Intrinsic reward weight
        self.w_alpha_decay = False

        self.w_beta_start = 0.001  # Worker entropy decay
        self.w_beta_decay = False

        # # Worker:
        self.w_lstm_size = self.m_lstm_size
        self.w_act_dim = 128
        self.w_discount = 0.9

    def randomize(self):
        min_logU = 10 ** -5.5
        max_logU = 10 ** -4.5

        min_logU_beta = 10 ** -4
        max_logU_beta = 10 ** -2

        self.learning_rate = logU(min_logU, max_logU)
        self.m_beta_start = logU(min_logU_beta, max_logU_beta)
        self.w_beta_start = logU(min_logU_beta, max_logU_beta)

        self.m_alpha_start = random.uniform()
        self.w_alpha_start = random.uniform()

        self.eta = random.uniform()
        self.m_eta = random.uniform()
        self.sm_eta = random.uniform()

        self.w_lstm_size = self.m_lstm_size = self.sm_lstm_size = self.sm_percept_dim = \
            self.sm_goal_dim = self.m_percept_dim = self.m_goal_dim = int(random.choice([64, 128, 256]))

        self.k = int(random.choice([8, 12, 16]))
