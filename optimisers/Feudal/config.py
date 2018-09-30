from optimisers.config import OptConfig, logU
from numpy import random


class FeudalConfig(OptConfig):
    def __init__(self, FLAGS, cluster):
        super().__init__(FLAGS, cluster)
        self.max_t = 300
        self.steps_in_ep = 6
        self.decay_steps = self.max_global_eps * self.steps_in_ep

        # Feudal:
        self.k = 16  # Dimensionality of w
        self.z_dim = 256  # Percept dimension
        self.nan_epsilon = 1e-8  # To prevent nans

        # # Manager:
        self.manager_lstm_size = 256
        self.manager_discount = 0.999
        self.manager_eta = 0.5

        self.s_dim = 256    # Manager percept dimension
        self.g_dim = 256    # Manager goal dimension
        self.c = 40         # Manager horizon

        self.random_goals = True
        self.g_eps_start = 0.005
        self.g_eps_end = 0
        self.g_eps_decay = True

        # # Worker:
        self.worker_lstm_size = self.manager_lstm_size
        self.worker_acts = 128
        self.worker_discount = 0.99

        self.alpha_start = 0.8  # Intrinsic reward weight
        self.alpha_end = 0.8
        self.alpha_decay = False

    def randomize(self):
        min_logU = 10 ** -6
        max_logU = 10 ** -4.5

        min_logU_beta = 10 ** -4
        max_logU_beta = 10 ** -2

        self.learning_rate = logU(min_logU, max_logU)
        self.beta_start = logU(min_logU_beta, max_logU_beta)
        self.alpha_start = random.uniform()
        self.eta = random.uniform()
        self.manager_eta = random.uniform()
        self.worker_lstm_size = self.manager_lstm_size = self.s_dim = self.g_dim = int(random.choice([256]))
        self.k = int(random.choice([16]))
