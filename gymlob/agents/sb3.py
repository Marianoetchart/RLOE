from gym import Env
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3

from omegaconf.dictconfig import DictConfig


class SB3Agent:

    def __init__(self,
                 env: Env,
                 cfg: DictConfig):
        self.env = env
        self.cfg = cfg
        self.model = None

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space

    def train(self):

        self.model = DDPG(policy='MlpPolicy',
                          env=self.env,
                          seed=self.cfg.seed,
                          verbose=1).learn(total_timesteps=self.cfg.num_train_episodes)

    def test(self):

        observation = self.env.reset()
        for i_episode in range(1, self.cfg.num_test_episodes + 1):

            action, _ = self.model.predict(observation)
            observation, reward, done, info = self.env.step(action)
            print("test episode: ", i_episode)
            print("action: ", action)
            print("observation: ", observation)
            print("reward: ", reward)

            if done:
                observation = self.env.reset()
