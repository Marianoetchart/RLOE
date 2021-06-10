#!/usr/bin/env bash

python scripts/train.py algo=dqn > dqn_train.out &
#python scripts/train.py algo=ddpg > ddpg_train.out &