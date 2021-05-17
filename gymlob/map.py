from gymlob.agents.dqn import DQNAgent
from gymlob.agents.ddpg import DDPGAgent
from gymlob.agents.ddpgfd import DDPGfDAgent

from gymlob.learners.dqn import DQNLearner
from gymlob.learners.ddpg import DDPGLearner
from gymlob.learners.ddpgfd import DDPGfDLearner

AGENT_MAPPING = {
    "DQN": DQNAgent,
    "DDPG": DDPGAgent,
    "DDPGfD": DDPGfDAgent,
}

LEARNER_MAPPING = {
    "DQN": DQNLearner,
    "DDPG": DDPGLearner,
    "DDPGfD": DDPGfDLearner,
}