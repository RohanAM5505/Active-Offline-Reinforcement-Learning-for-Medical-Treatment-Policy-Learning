import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from agent.offline_qlearning import OfflineQLearning
from agent.active_sampler import select_active_batch

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# ---------- BASELINE ----------
baseline_agent = OfflineQLearning()
baseline_rewards = []

for epoch in range(30):
    total = 0
    for s, a, r, s2, done in dataset:
        baseline_agent.update(s, a, r, s2, done)
        total += r
    baseline_rewards.append(total)

# ---------- ACTIVE ----------
active_agent = OfflineQLearning()
active_rewards = []

for round_id in range(5):
    batch = select_active_batch(dataset, active_agent, k=1000)
    total = 0

    for epoch in range(6):
        for s, a, r, s2, done in batch:
            active_agent.update(s, a, r, s2, done)
            total += r

    active_rewards.append(total)

# ---------- PLOT ----------
plt.figure()
plt.plot(baseline_rewards, label="Baseline Offline RL")
plt.plot(range(0, 30, 6), active_rewards, marker="o", label="Active Offline RL")
plt.xlabel("Training Steps")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.title("Baseline vs Active Offline RL")
plt.show()
