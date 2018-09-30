import threading
import queue


class RunnerThread(threading.Thread):
    def __init__(self, env, policy, num_local_steps, rollout_provider, result_tracker=None):
        super().__init__()
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.policy = policy
        self.rollout_provider = rollout_provider
        self.daemon = True
        self.sess = None
        self.result_tracker = result_tracker

    def start_runner(self, sess):
        self.sess = sess
        self.start()

    def run(self):
        with self.sess._tf_sess().as_default():
            self._run()

    def _run(self):
        rollout_provider = self.rollout_provider(self.env, self.policy, self.num_local_steps, self.result_tracker)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.
            self.queue.put(next(rollout_provider), timeout=600.0)
