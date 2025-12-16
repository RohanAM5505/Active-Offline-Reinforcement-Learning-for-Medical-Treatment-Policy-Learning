import numpy as np

# Actions
# 0 = no treatment
# 1 = low dose
# 2 = high dose
# 3 = lifestyle
ACTIONS = [0, 1, 2, 3]

def transition(state, action):
    """
    state: numpy array [HR, BP, Sugar, O2, Severity]
    returns next_state
    """
    next_state = np.array(state)

    # Only treatments can improve severity
    if action in [1, 2, 3]:
        if np.random.rand() < 0.6:  # probability of improvement
            next_state[4] = max(0, state[4] - 1)

    return next_state

def reward(state, next_state, action):
    """
    Medical reward design
    """
    r = -1  # base cost

    if next_state[4] < state[4]:
        r += 5  # improvement

    if next_state[4] == 0:
        r += 20  # recovery

    if action == 2:
        r -= 2  # high-dose risk

    return r
