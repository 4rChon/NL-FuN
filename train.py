from absl import app
from absl import flags
import os
import sys
from shlex import quote

flags.DEFINE_integer("num_workers", 8, "Number of workers.")
flags.DEFINE_integer("num_ps", 1, "Number of workers.")
flags.DEFINE_string("network", "FullyConv", "FullyConv, FullyConvLSTM, AtariNet.")
flags.DEFINE_string("optimiser", "a3c", "Name of the optimiser to use: a3c, feudal, deep_feudal.")
flags.DEFINE_string('log_path', "log", "Log directory path.")
flags.DEFINE_string('bat_path', ".", "Batch file directory path.")
flags.DEFINE_bool('linux', False, "Whether to output bat or sh.")
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("profiling", False, "Run with profiler.")
flags.DEFINE_string("profiling_file", "profile_output", "Profiling directory path.")
flags.DEFINE_integer("verbose", 3, "Terminal verbosity: 0 - no output; 1 - output commands; 2 - output notes; 3 - \
                                   output logs to terminal.")
flags.DEFINE_string("session", "testing", "Session name. Used for naming purposes.")
flags.DEFINE_string("python_v", "python", "Command to call python with.")

args = flags.FLAGS
args(sys.argv)


def new_cmd(session, name, cmd, log_path):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(quote(str(v)) for v in cmd)

    if args.verbose >= 3 and not args.linux:
        return name, "{}".format(cmd)
    if args.verbose >= 3 and args.linux:
        return name, "{} & echo kill $! >>{}/kill.sh".format(cmd, log_path)
    if args.linux:
        return name, "{} >{}/{}.{}.out 2>&1".format(cmd, log_path, session, name)
    return name, "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, log_path, session, name, log_path)


def create_commands(session, map_name, num_workers, num_ps, network, optimiser, log_path="/"):
    base_cmd = [
        'worker.py',
        '--map', map_name,
        '--num_workers', str(num_workers),
        '--num_ps', str(num_ps),
        '--log_path', log_path
    ]

    if args.profiling:
        base_cmd = [args.python_v,
                    '-m', 'cProfile',
                    '-o', args.profiling_file,
                    '-s', 'time'] + base_cmd

        f = open(args.profiling_file, 'wb')
        f.close()
    else:
        base_cmd = [args.python_v] + base_cmd

    cmds_map = []
    # start parameter server
    for i in range(num_ps):
        cmds_map += [new_cmd(session, "ps", base_cmd + ["--job_name", "ps", "--task", str(i)], log_path)]

    # start workers
    for i in range(num_workers):
        cmds_map += [new_cmd(
            session=session,
            name="w-%d" % i,
            cmd=base_cmd + ["--job_name", "worker", "--task", str(i), '--network', network, '--optimiser', optimiser],
            log_path=log_path
        )]

    # start tensorboard
    # ...

    cmds = [cmd for _, cmd in cmds_map]
    notes = []
    notes += ["Use `tail -f {}/*.out` to watch process output on linux or use --verbose 3.".format(log_path)]
    return cmds, notes


def write_to_file(cmds, linux=False):
    ext = 'sh' if linux else 'bat'

    for i in range(len(cmds)):
        with open(os.path.join(args.bat_path, 'run_{}.{}'.format(i, ext)), 'w') as bat:
            if not linux:
                bat.writelines([cmds[i], '\npause'])
            else:
                bat.writelines([cmds[i]])


def show_notes(notes):
    if args.verbose > 1:
        if len(notes) == 0:
            print("No notes available.")
        else:
            [print(note) for note in notes]


def _main(_):
    if args.optimiser == "feudal":
        args.network = "Feudal"
    if args.optimiser == "deep_feudal":
        args.network = "DeepFeudal"
    args.log_path = os.path.join(args.log_path, "{}_{}".format(args.optimiser, args.network))
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    cmds, notes = create_commands(args.session, args.map, args.num_workers, args.num_ps, args.network, args.optimiser, args.log_path)

    show_notes(notes)
    write_to_file(cmds, args.linux)


if __name__ == "__main__":
    app.run(_main)
