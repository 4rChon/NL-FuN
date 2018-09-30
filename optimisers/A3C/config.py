from optimisers.config import OptConfig, logU
from numpy import random


class A3CConfig(OptConfig):
    def __init__(self, FLAGS, cluster):
        super().__init__(FLAGS, cluster)
        self.max_t = 40
        self.steps_in_ep = 6
        self.decay_steps = self.max_global_eps * self.steps_in_ep

    def randomize(self):
        min_logU = 10 ** -3
        max_logU = 10 ** -2

        min_logU_beta = 10 ** -4
        max_logU_beta = 10 ** -2

        self.learning_rate = logU(min_logU, max_logU)
        self.beta_start = logU(min_logU_beta, max_logU_beta)
        self.eta = random.uniform()
