import omegaconf
import wandb


def set_wandb(cfg: omegaconf.dictconfig.DictConfig):
    wandb.init(project=cfg.project_name, name=cfg.algo.agent_type, reinit=True)
    wandb.config.update(cfg, allow_val_change=True)