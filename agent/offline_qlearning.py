import numpy as np

class OfflineQLearning:
    def __init__(self, alpha=0.05, gamma=0.95):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(4)
        return self.Q[state]

    def update(self, s, a, r, s2, done):
        q_sa = self.get_q(s)

        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.get_q(s2))

        q_sa[a] += self.alpha * (target - q_sa[a])

        # Conservative clipping (important for offline RL)
        q_sa[a] = np.clip(q_sa[a], -20, 20)
