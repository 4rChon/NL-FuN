from pysc2.env import sc2_env


def create_env(args):
    return sc2_env.SC2Env(map_name=args.map,
                          agent_race=args.bot_race,
                          difficulty=args.difficulty,
                          step_mul=args.step_mul,
                          screen_size_px=(args.screen_resolution, args.screen_resolution),
                          minimap_size_px=(args.minimap_resolution, args.minimap_resolution),
                          save_replay_episodes=args.save_replay_frequency,
                          replay_dir=args.save_dir,
                          visualize=args.render)
