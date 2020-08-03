import gym


class BaseEnv(gym.Env):
    """OpenAI Gym Limit Order Book Environment inspired from
    https://github.com/filangel/qtrader/blob/master/qtrader/envs/base.py

    Attributes
    ----------

    Methods
    -------
    step(action)
        The agent takes a step in the environment
    reset()
        Resets the state of the environment and returns an initial observation
    render()
        Present real-time data on a dashboard
    register(agent)
        Add an agent to the environment
    unregister(agent)
        Remove an agent from the environment
    """

    def __init__(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError