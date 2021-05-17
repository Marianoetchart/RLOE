#!/usr/bin/env bash

nohup python scripts/train.py algo=dqn > dqn_train.out &
nohup python scripts/train.py algo=ddpg > ddpg_train.out &