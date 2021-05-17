ROOT_DIR = '/efs/mm/gymlob/'
#ROOT_DIR = '/Users/mahmoud/_code/gymlob/'

import sys
import os
from joblib import Parallel, delayed
import hydra
from omegaconf import OmegaConf, DictConfig

os.environ["WANDB_START_METHOD"] = "thread"
sys.path.append(ROOT_DIR)

from gymlob.map import AGENT_MAPPING
from gymlob.envs.l2_market_replay_env import L2MarketReplayEnv
from gymlob.utils.utils import set_seed
from gymlob.utils.logging import set_wandb

CONFIG_NAME = ROOT_DIR + "configs/ddpg_agent.yaml"
CONFIG_PATH = ROOT_DIR + 'configs/'


@hydra.main(config_name=CONFIG_NAME, config_path=CONFIG_PATH)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    Parallel(n_jobs=cfg.num_experiments_in_parallel,
             backend='multiprocessing')(delayed(run)(cfg.seeds[i], cfg)
                                        for i in range(len(cfg.seeds)))


def run(seed: int,
        cfg: DictConfig):

    env = L2MarketReplayEnv(instrument=cfg.instrument,
                            date=cfg.date,
                            frequency=cfg.frequency,
                            num_levels=10,
                            client_order_info={
                                "direction": cfg.direction,
                                "quantity": cfg.quantity,
                                "duration": cfg.duration,
                                "benchmark": cfg.benchmark
                            },
                            orderbook_file_path=cfg.orderbook_file_path,
                            orders_file_path=cfg.orders_file_path,
                            _seed=seed)

    set_seed(environment=env, seed=seed)

    agent = AGENT_MAPPING.get(cfg.algo.agent_type)(env, cfg)
    if cfg.log_wb:
        set_wandb(cfg)
    agent.train()


if __name__ == "__main__":
    main()
