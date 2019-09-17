from pysc2.env import sc2_env


def create_env(args):
    agent = sc2_env.Agent(sc2_env.Race.protoss, args.optimiser + "_" + args.network)

    agent_interface_format = sc2_env.parse_agent_interface_format(
        feature_screen=[args.screen_resolution, args.screen_resolution],
        feature_minimap=[args.minimap_resolution, args.minimap_resolution],
        rgb_screen=[args.screen_resolution, args.screen_resolution],
        rgb_minimap=[args.minimap_resolution, args.minimap_resolution],
        action_space="FEATURES"
    )

    return sc2_env.SC2Env(
        map_name=args.map,
        players=[agent],
        agent_interface_format=[agent_interface_format],
        step_mul=args.step_mul,
        save_replay_episodes=args.save_replay_frequency,
        replay_dir=args.save_dir,
        visualize=args.render
    )

