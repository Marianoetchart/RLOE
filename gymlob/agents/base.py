from abc import abstractmethod


class Agent:

    _id = 'base'

    def __init__(self, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self._id

    @abstractmethod
    def act(self, observation):
        raise NotImplementedError