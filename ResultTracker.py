import csv
import os


class ResultTracker:
    def __init__(self, logdir, result_file="results"):
        self.logdir = logdir
        self.result_file = result_file
        self._init_directory()

    def _init_directory(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def save(self, time, reward):
        with open(os.path.join(self.logdir, self.result_file + '.csv'), 'a', newline='') as csv_file:
            w = csv.writer(csv_file, delimiter=',')
            w.writerow([time, reward])
