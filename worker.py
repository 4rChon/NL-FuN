import tensorflow as tf
import logging
import os
from pysc2.env import sc2_env
import time

from optimisers import init_optimiser, init_config
from envs import create_env
from ResultTracker import ResultTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tf.flags.DEFINE_integer('task', 0, 'Task index')
tf.flags.DEFINE_string('job_name', "worker", 'worker or ps')
tf.flags.DEFINE_integer('num_ps', 1, 'Number of workers')
tf.flags.DEFINE_integer('num_workers', 1, 'Number of workers')
tf.flags.DEFINE_bool('local', True, '')
tf.flags.DEFINE_string('log_path', "log", 'Log directory path')
tf.flags.DEFINE_string("device", "cpu", "Device to use for tensor operations.")

# Environment Flags
tf.flags.DEFINE_string("map", "DefeatSingleZealot", "Name of a map to use.")
tf.flags.DEFINE_enum("agent_race", "T", sc2_env.races.keys(), "Agent's race.")
tf.flags.DEFINE_enum("bot_race", "T", sc2_env.races.keys(), "Opponent's race.")
tf.flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
tf.flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
tf.flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
tf.flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
tf.flags.DEFINE_integer("save_replay_frequency", 1000000, "Replay saving frequency.")
tf.flags.DEFINE_string("save_dir", "replays/", "Directory where replays will be saved.")
tf.flags.DEFINE_bool("render", False, "Whether to render pysc2 visualisation")

tf.flags.DEFINE_string("network", "FullyConv", "Name of the network to use: FullyConv, FullyConvLSTM, AtariNet")
tf.flags.DEFINE_string("optimiser", "a3c", "Name of the optimiser to use: a3c, feudal.")

# Training envs params
tf.flags.DEFINE_bool("continuation", False, "Whether to load a previous checkpoint.")
tf.flags.DEFINE_bool("is_training", True, "Whether to train the agent.")

FLAGS = tf.flags.FLAGS


def get_dir(path, map_name, name):
    directory = os.path.join(path, map_name, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def get_dir_index(path, map_name):
    i = 0
    while os.path.exists(os.path.join(path, map_name + "_" + str(i))):
        i += 1

    if FLAGS.task == 0 and not FLAGS.continuation:
        directory = os.path.join(path, map_name+ "_" + str(i))
        os.makedirs(directory)
    else:
        i -= 1

    return i


def run(server, cluster):
    FLAGS.log_path = FLAGS.log_path.strip('\'')
    index = get_dir_index(FLAGS.log_path, FLAGS.map)
    checkpoint_dir = get_dir(FLAGS.log_path, FLAGS.map + "_" + str(index), 'checkpoints')
    summary_dir = get_dir(FLAGS.log_path, FLAGS.map + "_" + str(index), 'summaries')

    summary_writer = tf.summary.FileWriter(summary_dir) if FLAGS.task == 0 else None

    env = create_env(FLAGS)
    Config = init_config(FLAGS.optimiser)
    opt_config = Config(FLAGS, cluster)

    # Set up result path and randomize hyperparams
    results_dir = get_dir(FLAGS.log_path, FLAGS.map + "_" + str(index), 'results')
    result_tracker = ResultTracker(results_dir, result_file="results_{}".format(FLAGS.task))
    if FLAGS.task == 0 and not FLAGS.continuation and FLAGS.is_training:
        opt_config.randomize()
        opt_config.save(results_dir)
    else:
        opt_config.load(results_dir)

    # Build graphs
    optimiser = init_optimiser(FLAGS, FLAGS.task, env, opt_config, summary_writer, result_tracker)
    if FLAGS.task == 0:
        for task in range(1, FLAGS.num_workers):
            init_optimiser(FLAGS, task, env, opt_config)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('\t%s %s', v.name, v.get_shape())

    variables = [v for v in tf.global_variables() if not v.name.startswith("global")]

    init_all_op = tf.global_variables_initializer()

    def init_fn(self, sess):
        logger.info("Initializing parameters")
        sess.run(init_all_op)

    init_op = tf.variables_initializer(variables)
    ready_op = tf.report_uninitialized_variables(variables)

    scaffold = tf.train.Scaffold(
        init_fn=init_fn,
        init_op=init_op,
        ready_op=ready_op,
    )

    config = tf.ConfigProto(
        device_filters=[
            "/job:ps",
            "/job:worker/task:{}/{}:{}".format(FLAGS.task, "cpu", optimiser.cpu_ix),
            "/job:worker/task:{}/{}:{}".format(FLAGS.task, "gpu", optimiser.gpu_ix)
        ]
    )

    stop_hook = tf.train.StopAtStepHook(num_steps=opt_config.max_global_eps*opt_config.steps_in_ep)
    # sync_replicas_hook = optimiser.opt.make_session_run_hook(args.task == 0)
    hooks = [
        stop_hook,
        # sync_replicas_hook,
    ]

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified."
    )

    # Summaries set to none as some summaries require part of the graph to be populated already
    # Otherwise we get a placeholder feed dict error
    start = time.time()
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task == 0),
                                           checkpoint_dir=checkpoint_dir,
                                           scaffold=scaffold,
                                           hooks=hooks,
                                           config=config,
                                           save_checkpoint_steps=50,
                                           save_summaries_steps=None,
                                           save_summaries_secs=None) as mon_sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        global_step_init = -1
        if ckpt and ckpt.model_checkpoint_path:
            global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

        # Start the RunnerThread
        optimiser.start(mon_sess)
        mon_sess.run(optimiser.sync)
        # Start or Continue training
        if FLAGS.is_training:
            logger.info("Starting training at step = {}".format(global_step_init))
        while not mon_sess.should_stop():
            # Process a rollout
            step_op = tf.train.get_or_create_global_step()
            step = mon_sess.run(step_op)
            if FLAGS.is_training:
                logger.info("Next rollout: {}".format(step))
                optimiser.process(mon_sess)

    logger.info("Events directory: %s", summary_dir)

    # Ask for all the services to stop
    end = time.time() - start
    logger.info('reached {} episodes after {} seconds. worker stopped.'.format(opt_config.max_global_eps, end))


def cluster_spec(num_workers, num_ps):
    cluster = {'ps': [], 'worker': []}
    port = 12222

    host = 'localhost'
    for _ in range(num_ps):
        cluster['ps'].append('{}:{}'.format(host, port))
        port += 1

    for _ in range(num_workers):
        cluster['worker'].append('{}:{}'.format(host, port))
        port += 1

    return cluster


def main(unused_argv):
    spec = cluster_spec(FLAGS.num_workers, FLAGS.num_ps)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(spec)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        # device_count={"CPU": 8, "GPU": 2},
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    # Create and start a server for the local task
    # with tf.device("/CPU:{}".format(FLAGS.task % 8)):
    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task,
        config=config
    )

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        run(server, spec)


if __name__ == "__main__":
    tf.app.run(main)
