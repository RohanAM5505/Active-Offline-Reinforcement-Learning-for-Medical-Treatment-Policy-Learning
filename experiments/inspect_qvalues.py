import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from agent.offline_qlearning import OfflineQLearning

# load dataset
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# train agent quickly (active-style)
agent = OfflineQLearning()

for epoch in range(30):
    for s, a, r, s2, done in dataset:
        agent.update(s, a, r, s2, done)

# pick a sample patient
sample_state = dataset[0][0]

q_vals = agent.get_q(sample_state)

actions = ["no_treatment", "low_dose", "high_dose", "lifestyle"]

print("Sample patient state:", sample_state)
print("Learned Q-values:")
for a, q in zip(actions, q_vals):
    print(f"{a:15s}: {q:.2f}")

print("\nRecommended action:", actions[int(np.argmax(q_vals))])
