import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
from agent.offline_qlearning import OfflineQLearning
from agent.active_sampler import select_active_batch

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

agent = OfflineQLearning()

for round_id in range(5):
    active_batch = select_active_batch(dataset, agent, k=1000)

    for epoch in range(20):
        for s, a, r, s2, done in active_batch:
            agent.update(s, a, r, s2, done)

    print(f" Active training round {round_id + 1} completed")
