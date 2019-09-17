import logging
from optimisers import DeepFeudal, Feudal, A3C, config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_optimiser(args, task, env, config, summary_writer=None, result_tracker=None):
    if args.optimiser == 'a3c':
        return A3C.optimiser(
            env=env,
            config=config,
            task=task,
            network=args.network,
            summary_writer=summary_writer,
            result_tracker=result_tracker
        )
    elif args.optimiser == 'feudal':
        return Feudal.optimiser(
            env=env,
            config=config,
            task=task,
            network=args.network,
            summary_writer=summary_writer,
            result_tracker=result_tracker
        )
    elif args.optimiser == 'deep_feudal':
        return DeepFeudal.optimiser(
            env=env,
            config=config,
            task=task,
            network=args.network,
            summary_writer=summary_writer,
            result_tracker=result_tracker
        )
    else:
        logger.error("Invalid Architecture. Required 'a3c' or 'feudal', found {} instead.".format(args.arch))
        exit(0)


def init_config(opt):
    if opt == 'a3c':
        return A3C.config
    if opt == 'feudal':
        return Feudal.config
    if opt == 'deep_feudal':
        return DeepFeudal.config
