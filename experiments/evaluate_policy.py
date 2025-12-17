import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from agent.offline_qlearning import OfflineQLearning
from agent.active_sampler import select_active_batch

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Train baseline agent
baseline = OfflineQLearning()
for epoch in range(30):
    for s, a, r, s2, done in dataset:
        baseline.update(s, a, r, s2, done)

# Train active agent
active = OfflineQLearning()
for _ in range(5):
    batch = select_active_batch(dataset, active, k=1000)
    for epoch in range(20):
        for s, a, r, s2, done in batch:
            active.update(s, a, r, s2, done)

def policy_agreement(agent, data):
    correct = 0
    for s, a, _, _, _ in data[:500]:
        if np.argmax(agent.get_q(s)) == a:
            correct += 1
    return correct / 500

print("Baseline policy agreement:", policy_agreement(baseline, dataset))
print("Active policy agreement  :", policy_agreement(active, dataset))
