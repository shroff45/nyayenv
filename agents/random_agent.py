class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.action_space.sample()

    def __repr__(self):
        return "RandomAgent"
