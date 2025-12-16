# Active Offline Reinforcement Learning for Medical Treatment Policy Learning

## Overview
This project implements a **standalone Active + Offline Reinforcement Learning (RL) framework** for learning medical treatment policies from **historical (logged) patient data**.

The work is inspired by challenges in **safe decision-making**, where online exploration is infeasible (e.g., healthcare), and by ideas from **Active Reinforcement Learning**, which aim to improve learning efficiency by selectively focusing on informative data.

Importantly, this project studies **both the strengths and limitations** of naive uncertainty-based active sampling in offline medical RL.

---

## Motivation
Reinforcement Learning is naturally suited for **sequential clinical decision-making**, but direct interaction with real patients is unsafe and unethical. This motivates **Offline RL**, where policies are learned solely from historical data.

At the same time, medical data is expensive and scarce, motivating **Active RL**, which asks:
> *Which data points are most useful for improving the policy?*

This project explores the intersection of these two ideas.

---

## Problem Formulation
The medical decision process is modeled as a **Markov Decision Process (MDP)**:

- **State**: Patient vitals and disease severity  
  `[heart rate, blood pressure, blood sugar, oxygen level, severity]`
- **Actions**:  
  `no treatment`, `low-dose medication`, `high-dose medication`, `lifestyle advice`
- **Reward**: Encodes health improvement, recovery, treatment cost, and risk
- **Objective**: Learn a treatment policy that maximizes **long-term expected health outcomes**

---

## Methodology

### 1. Offline Reinforcement Learning
- Q-learning with Bellman updates
- Learning exclusively from logged patient transitions
- Conservative value clipping to prevent unsafe extrapolation

### 2. Active Data Selection
- Data selection based on **Q-value uncertainty**
- High-uncertainty states are prioritized for training
- No online interaction or exploration is performed

### 3. Baseline vs Active Comparison
Two training regimes are compared:
- **Baseline Offline RL**: Random subsets of logged data
- **Active Offline RL**: Uncertainty-selected subsets of the same size

---

## Key Findings
- **Policy agreement with logged clinician actions saturates quickly**, making it an unreliable primary metric.
- **Naive uncertainty-based active sampling does not consistently outperform baseline offline RL** in terms of expected policy value.
- In some settings, active sampling can **slightly degrade performance** due to distributional bias.

These results highlight a critical insight:
> *Active RL in offline, safety-critical domains requires careful, distribution-aware design.*

---

## Project Structure
