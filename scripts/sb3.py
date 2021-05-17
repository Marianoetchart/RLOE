ROOT_DIR = '/efs/mm/gymlob/'
# ROOT_DIR = '/Users/mahmoud/_code/gymlob/'

import sys

import hydra
from omegaconf import OmegaConf, DictConfig

sys.path.append(ROOT_DIR)
from gymlob.agents.sb3 import SB3Agent
from gymlob.envs.l2_market_replay_env import L2MarketReplayEnv

CONFIG_NAME = ROOT_DIR + "configs/test_config.yaml"
CONFIG_PATH = ROOT_DIR + 'configs/'


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    env = L2MarketReplayEnv(instrument=cfg.instrument,
                            date=cfg.date,
                            frequency=cfg.frequency,
                            num_levels=10,
                            client_order_info={"direction": cfg.direction,
                                               "quantity": cfg.quantity,
                                               "duration": cfg.duration,
                                               "benchmark": cfg.benchmark},
                            orderbook_file_path=cfg.orderbook_file_path,
                            orders_file_path=cfg.orders_file_path,
                            discrete_action_space=True,
                            _seed=cfg.seed)

    agent = SB3Agent(env=env, cfg=cfg)

    agent.train()
    agent.test()


if __name__ == "__main__":
    main()
