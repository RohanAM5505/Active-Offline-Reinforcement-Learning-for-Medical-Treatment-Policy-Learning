import numpy as np

def uncertainty(agent, state):
    q_values = agent.get_q(state)
    return np.std(q_values)

def select_active_batch(dataset, agent, k=1000):
    scored = []

    for transition in dataset:
        state = transition[0]
        score = uncertainty(agent, state)
        scored.append((score, transition))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [t for _, t in scored[:k]]
