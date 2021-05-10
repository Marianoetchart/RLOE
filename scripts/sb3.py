import sys

sys.path.append('/Users/mahmoud/_code/gymlob/')

import hydra
from omegaconf import OmegaConf, DictConfig

from gymlob.agents.sb3 import SB3Agent
from gymlob.envs.market_replay_env import MarketReplayEnv

CONFIG_NAME = "/Users/mahmoud/_code/gymlob/configs/test_config.yaml"
CONFIG_PATH = '/Users/mahmoud/_code/gymlob/configs/'


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    env = MarketReplayEnv(instrument=cfg.instrument,
                          date=cfg.date,
                          frequency=cfg.frequency,
                          num_levels=10,
                          client_order_info={"direction": cfg.direction,
                                             "quantity": cfg.quantity,
                                             "duration": cfg.duration,
                                             "benchmark": cfg.benchmark},
                          orderbook_file_path=cfg.orderbook_file_path,
                          orders_file_path=cfg.orders_file_path,
                          _seed=cfg.seed)

    agent = SB3Agent(env=env,
                     cfg=cfg)

    agent.train()
    agent.test()


if __name__ == "__main__":
    main()