import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
from agent.offline_qlearning import OfflineQLearning

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

agent = OfflineQLearning()
subset = dataset[:1000] 
for epoch in range(30):
    for s, a, r, s2, done in dataset:
        agent.update(s, a, r, s2, done)

print("Baseline offline RL training complete")
