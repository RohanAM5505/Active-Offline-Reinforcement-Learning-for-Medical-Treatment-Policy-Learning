import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
from mdp.mdp import transition, reward, ACTIONS

def random_state():
    return np.array([
        np.random.randint(60, 100),   # Heart rate
        np.random.randint(90, 140),   # Blood pressure
        np.random.randint(100, 200),  # Sugar
        np.random.randint(90, 100),   # Oxygen
        np.random.randint(1, 4)       # Severity (1=mild, 3=severe)
    ])

dataset = []

for _ in range(6000):
    state = random_state()
    severity = state[4]

    # -------- CLINICIAN BEHAVIOR POLICY --------
    # mild → low-dose or lifestyle
    if severity == 1:
        action = np.random.choice([1, 3], p=[0.7, 0.3])

    # moderate → low-dose or high-dose
    elif severity == 2:
        action = np.random.choice([1, 2], p=[0.6, 0.4])

    # severe → high-dose mostly
    else:  # severity == 3
        action = np.random.choice([2, 1], p=[0.7, 0.3])
    # -------------------------------------------

    next_state = transition(state, action)
    r = reward(state, next_state, action)
    done = next_state[4] == 0

    dataset.append(
        (tuple(state), action, r, tuple(next_state), done)
    )

with open("dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("✅ Offline medical dataset created with clinician behavior (dataset.pkl)")
