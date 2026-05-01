# rl-autonomous-vehicles
Reinforcement learning for autonomous vehicle control in MetaDrive and ScenarioNet
# Deep Reinforcement Learning for Autonomous Vehicle Control

This repository contains the implementation of deep reinforcement learning algorithms for training and evaluating autonomous vehicle agents in simulation environments. The goal is to benchmark multiple RL approaches and identify the most effective algorithm for autonomous vehicle control and decision-making.

---

## Overview

Autonomous vehicles require robust decision-making policies to handle complex, dynamic driving environments. This project applies and compares state-of-the-art deep RL algorithms — training agents in realistic simulation environments and evaluating their performance across standard and safety-critical scenarios.

---

## Algorithms Implemented

| Algorithm | Full Name | Type |
|-----------|-----------|------|
| TD3 | Twin Delayed Deep Deterministic Policy Gradient | Off-policy, Continuous Action |
| PPO | Proximal Policy Optimization | On-policy, Continuous Action |

*Planned: DQN (Deep Q-Network) and DDPG (Deep Deterministic Policy Gradient) — to be implemented for benchmarking against TD3 and PPO.*

---

## Simulation Environments

- **ScenarioNet** — Primary training environment. Used for structured scenario-based training and agent evaluation across diverse driving conditions.
- **MetaDrive** — Used for safety-critical scenario testing, including lane-changing and obstacle avoidance under diverse driving conditions.

Both environments are built on top of realistic traffic simulation and support configurable road networks, traffic agents, and scenario complexity.

---

## Project Structure

```
rl-autonomous-vehicles/
│
├── td3/                  # TD3 training and evaluation scripts
├── ppo/                  # PPO training and evaluation scripts
├── envs/                 # Environment configurations (ScenarioNet, MetaDrive)
├── utils/                # Reward functions, logging, helper utilities
└── results/              # Training logs, reward curves, evaluation metrics
```

---

## Key Features

- **Reward Function Redesign** — Custom reward shaping applied to TD3 training to optimize for smooth acceleration profiles and reduced steering variance.
- **Hyperparameter Tuning** — Systematic tuning of learning rates, discount factors, and replay buffer parameters across algorithms.
- **Safety-Critical Evaluation** — TD3 agent tested in MetaDrive safety scenarios to assess generalization beyond the training distribution.
- **Benchmarking Framework** — Unified training pipeline enabling fair, reproducible comparison across all four algorithms.

---

## Dependencies

- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)
- [MetaDrive](https://github.com/metadriverse/metadrive)
- Python 3.8+
- PyTorch

---

## Research Context

This work is conducted as part of ongoing graduate research in the **Emerging Mobility and Control Systems Lab** at Morgan State University, focusing on AI and machine learning applications for autonomous and connected vehicles.

---

## Status

🔬 Active Research — algorithms are being trained, evaluated, and compared. Results and final benchmarks will be updated as experiments are completed.
```
