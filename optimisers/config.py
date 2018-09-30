from numpy import exp, log, random
import json
import os


def logU(lo, hi):
    return exp(random.uniform(log(lo), log(hi)))


class OptConfig:
    def __init__(self, FLAGS, cluster):
        self.learning_rate_decay = True
        self.learning_rate = 1e-3
        self.max_global_eps = 1000
        self.use_batch_norm = True

        self.entropy_regularization = True
        self.beta_decay = False
        self.beta_start = .001
        self.beta_end = .0001

        self.eta = 0.5  # Value loss weight
        self.gamma = 0.99  # Discount rate for future rewards

        self.e_greedy = False
        self.epsilon_start = .5
        self.epsilon_end = .1
        self.epsilon_decay = True

        self.dropout_keep_prob = 1.0

        # Distributed:
        self.local = FLAGS.local
        self.num_ps = FLAGS.num_ps
        self.num_workers = FLAGS.num_workers
        self.cluster = cluster

    def randomize(self):
        pass

    def save(self, logdir, file="params"):
        with open(os.path.join(logdir, file + '.json'), 'a') as param_file:
            json.dump(self.__dict__, param_file)

    def load(self, logdir, file="params"):
        with open(os.path.join(logdir, file + '.json'), 'r') as param_file:
            obj = json.load(param_file)
            for v in obj:
                setattr(self, v, obj[v])






