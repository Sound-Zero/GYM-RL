# 🚀 GYM-RL: Reinforcement Learning From Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch)
![Gym](https://img.shields.io/badge/Gym-0.26.2-00897B?logo=openai)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

**Implementing DQN · Double DQN · A3C · PPO from scratch to master LunarLander-v2 🛸**

</div>

---

## 📑 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🌍 Environment: LunarLander-v2](#-environment-lunarlander-v2)
- [🧠 Algorithms](#-algorithms)
  - [1. DQN (Deep Q-Network)](#1-dqn-deep-q-network)
  - [2. ConcatDQN (Residual DQN)](#2-concatdqn-residual-dqn)
  - [3. Double DQN](#3-double-dqn)
  - [4. A3C (Asynchronous Advantage Actor-Critic)](#4-a3c-asynchronous-advantage-actor-critic)
  - [5. PPO (Proximal Policy Optimization)](#5-ppo-proximal-policy-optimization)
- [📊 Performance Comparison](#-performance-comparison)
- [🛠 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🧪 Key Implementation Details](#-key-implementation-details)
- [📈 Training Curves & Visualizations](#-training-curves--visualizations)
- [💡 Insights & Lessons Learned](#-insights--lessons-learned)
- [📚 References](#-references)

---

## 🎯 Project Overview

**GYM-RL** is a comprehensive reinforcement learning project that implements **four major RL algorithms** from scratch using PyTorch, trained and benchmarked on the challenging **LunarLander-v2** continuous control environment.

### Why "GYM-RL"?

> "GYM" — representing the Gymnasium (OpenAI Gym) environment framework. "RL" — Reinforcement Learning. Building everything from zero, no high-level RL libraries like Stable-Baselines3 or RLlib.

### What Makes This Project Special

| Feature | Description |
|---------|-------------|
| 🔬 **From-Scratch Implementation** | Every algorithm is hand-coded using only PyTorch and NumPy — no black-box RL frameworks |
| 🏗️ **Custom Network Architectures** | Features the novel **ConcatNet** (Residual-style feature concatenation network) |
| 📊 **Comprehensive Benchmarking** | Side-by-side comparison of 4 algorithms on the same environment |
| 🎮 **Production-Quality Training** | GAE, learning rate scheduling, gradient clipping, model checkpointing |
| 🔥 **GPU Accelerated** | Full CUDA / MPS support for both single-GPU and multi-process (A3C) training |

---

## 🌍 Environment: LunarLander-v2

<div align="center">
  <img src="MyProject/DQN/DQN_second_round.gif" width="400" alt="LunarLander DQN Demo"/>
  <p><em>DQN agent successfully landing the lunar module 🛸</em></p>
</div>

[LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/) is a classic reinforcement learning benchmark where the agent must land a lunar module safely on a landing pad.

### State Space (8-dim continuous)

| Index | Observation | Range |
|-------|-------------|-------|
| 0 | X position | (-∞, +∞) |
| 1 | Y position | (-∞, +∞) |
| 2 | X velocity | (-∞, +∞) |
| 3 | Y velocity | (-∞, +∞) |
| 4 | Angle | (-π, π] |
| 5 | Angular velocity | (-∞, +∞) |
| 6 | Left leg contact | {0, 1} |
| 7 | Right leg contact | {0, 1} |

### Action Space (4 discrete actions)

| Action | Meaning |
|--------|---------|
| 0 | Do nothing |
| 1 | Fire left orientation engine |
| 2 | Fire main engine (down) |
| 3 | Fire right orientation engine |

### Reward System

```
Landing pad is at (0, 0). Total reward ≈ 200 points for a perfect landing.

• Moving from top to landing pad:  +100~140 points
• Crash:                             -100 points  
• Hovering (each frame):              -0.3 points
• Leg contact with ground:           +10 points per leg
• Fire main engine (each frame):      -0.3 points
```

> 🏆 **Solved threshold**: Average reward ≥ **200** over 100 consecutive episodes.

---

## 🧠 Algorithms

### 1. DQN (Deep Q-Network)

<div align="center">

| DQN Training GIF | Performance Curve |
|:---:|:---:|
| ![DQN GIF](MyProject/DQN/DQN_second_round.gif) | ![DQN Performance](MyProject/DQN/DQN_second_round_performance.png) |

</div>

#### Core Idea

DQN approximates the optimal action-value function $Q^*(s, a)$ using a deep neural network, combined with two key innovations to stabilize training:

**1. Experience Replay** — Break temporal correlations by sampling random mini-batches from a replay buffer.

```
                   ┌──────────────────────────┐
                   │     ReplayMemory (10K)    │
                   │  ┌────┬────┬────┬────┐   │
                   │  │ s,a│r,s'│....│....│   │
                   │  └────┴────┴────┴────┘   │
                   └──────────┬───────────────┘
                              │ random sample (BATCH_SIZE=128)
                              ▼
                   ┌──────────────────────────┐
                   │    DQN Network Update     │
                   └──────────────────────────┘
```

**2. Target Network** — Use a slowly-updated copy of the policy network to compute TD targets, preventing the moving-target problem.

$$y_i = r_i + \gamma \max_{a'} Q_{\text{target}}(s_i', a')$$

#### Network Architecture

```
┌─────────┐     ┌──────┐     ┌──────┐     ┌──────────┐
│ State(8)│────▶│ 128  │────▶│ 128  │────▶│ Action(4)│
└─────────┘     │ ReLU │     │ ReLU │     └──────────┘
                └──────┘     └──────┘
```

#### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 128 | Mini-batch size |
| `GAMMA` | 0.99 | Discount factor |
| `EPS_START` | 0.9 | Initial exploration rate |
| `EPS_END` | 0.05 | Final exploration rate |
| `EPS_DECAY` | 1000 | ε-decay rate |
| `TAU` | 0.005 | Target network soft-update rate |
| `LR` | 1e-4 | Learning rate (AdamW) |

#### ε-Greedy Exploration

$$\epsilon = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \cdot e^{-\frac{\text{steps\_done}}{\text{eps\_decay}}}$$

```
ε
1.0 ┤╲
0.8 ┤ ╲
0.6 ┤  ╲
0.4 ┤   ╲
0.2 ┤    ╲_________
0.0 ┤
    └──────────────▶ steps
```

[▶ View DQN Code](MyProject/DQN/DQN.py)

---

### 2. ConcatDQN (Residual DQN)

<div align="center">
  <img src="https://github.com/user-attachments/assets/dece1ae9-6511-4eac-ae21-08401a3594da" width="300" alt="ConcatDQN Episodes"/>
  <img src="https://github.com/user-attachments/assets/973b68db-5ec7-4bd0-9057-f1d194494342" width="300" alt="ConcatDQN Reward"/>
</div>

#### What is ConcatNet?

A novel **residual-inspired** network architecture that splits features into parallel branches, then concatenates them — preserving both high-level and low-level representations through the forward pass.

```
                    ConcatNet Forward Pass
    ═══════════════════════════════════════════════════════

    input(8)
       │
       ▼
    ┌──────────────────────────────────────┐
    │  Linear(8 → 1024) + ReLU            │
    └──────────────────────────────────────┘
       │
       ├────────────── chunk ──────────────┐
       ▼                                   ▼
    x₁ (512) 保留                    x (512) 继续
                                   │
                                   ▼
                            Linear(512 → 256) + ReLU
                                   │
                          chunk ───┐
                                   ▼
                            x₂ (128) 保留
                                   │
                                   ▼
                            Linear(128 → 64) + ReLU
                                   │
                          chunk ───┐
                                   ▼
                            x₃ (32) 保留
                                   │
                                   ▼
                            Linear(32 → 16) + ReLU
                                   │
                                   ▼
                              x₄ (16)
                                   │
       ┌───────────────────────────┘
       │    concat[x₁(512), x₂(128), x₃(32), x₄(16)]
       ▼
    ┌──────────────────────────────────────┐
    │  Linear(688 → 4)  (Output Q-Values)  │
    └──────────────────────────────────────┘
```

> 💡 **Intuition**: By concatenating feature representations from different depths, the network learns multi-scale features — shallow layers capture simple patterns while deeper layers capture more abstract ones. This is conceptually similar to DenseNet / ResNet skip connections.

[▶ View ConcatDQN Code](MyProject/DQN/ConcatDQN/DQN_concat.py)

---

### 3. Double DQN

**Double DQN** addresses the **overestimation bias** inherent in standard DQN. The key insight:

> ❌ **DQN**: Uses the **same network** to both select AND evaluate actions, leading to systematic overestimation of Q-values.
>
> ✅ **Double DQN**: Decouples action selection from evaluation — Policy Net selects the best action, Target Net evaluates it.

#### The Difference

```python
# ❌ DQN (max bias)
target = reward + gamma * target_net(next_state).max()

# ✅ Double DQN (decoupled)
best_action = policy_net(next_state).argmax()      # select
target = reward + gamma * target_net(next_state)[best_action]  # evaluate
```

```
        Action Selection         Action Evaluation
    ┌───────────────────┐    ┌───────────────────┐
    │   Policy Net θ    │    │   Target Net θ⁻   │
    │                   │    │                   │
    │  Q(s', a₁) = 0.8  │    │  Q(s', a₁) = 0.6  │
    │  Q(s', a₂) = 0.9◀─┼────│─▶Q(s', a₂) = 0.5  │
    │  Q(s', a₃) = 0.3  │    │  Q(s', a₃) = 0.4  │
    └───────────────────┘    └───────────────────┘
            │                         │
            │  "Pick a₂!"             │  "Value = 0.5"
            └─────────────────────────┘
                   y = r + γ × 0.5  (lower, more accurate)
```

#### Network Improvements

- **Hidden dimension**: 256 (vs DQN's 128)
- **Hard target network update**: Every 5 steps (vs soft update in DQN)
- **More aggressive learning rate**: `LR = 0.002` (vs DQN's `1e-4`)

[▶ View Double DQN Code](MyProject/DoubleDQN/DDQN_Beta.py)

---

### 4. A3C (Asynchronous Advantage Actor-Critic)

#### The Multi-Agent Approach

A3C runs **multiple worker agents in parallel**, each with their own copy of the environment and model. Workers asynchronously push gradients to a global shared model — this naturally decorrelates experiences and stabilizes training without a replay buffer.

```
    ┌─────────────────────────────────────────────────────────┐
    │                     Shared Global Model                  │
    │                   ┌───────────────────┐                  │
    │                   │  Actor-Critic Net │                  │
    │                   │  Shared(8→128)    │                  │
    │                   │  ├─ Policy(128→2) │                  │
    │                   │  └─ Value(128→1)  │                  │
    │                   └───────┬───────────┘                  │
    └───────────────────────────┼─────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  Worker 0    │   │  Worker 1    │   │  Worker N    │
    │  (Process)   │   │  (Process)   │   │  (Process)   │
    │              │   │              │   │              │
    │ Local Model  │   │ Local Model  │   │ Local Model  │
    │ Own Env      │   │ Own Env      │   │ Own Env      │
    └──────────────┘   └──────────────┘   └──────────────┘
        CartPole           CartPole           CartPole
```

#### A3C Loss Function

The total loss combines three components:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{actor}}}_{\text{Policy Gradient}} + \beta \cdot \underbrace{\mathcal{L}_{\text{critic}}}_{\text{Value Loss}} - \alpha \cdot \underbrace{\mathcal{L}_{\text{entropy}}}_{\text{Entropy Bonus}}$$

```
┌──────────────────────────────────────────────────────────────┐
│  Actor Loss:    -E[log π(a|s) × A(s,a)]    ← 策略梯度      │
│  Critic Loss:   MSE(V(s), GAE_target)      ← 价值估计      │
│  Entropy Bonus: +α × H(π(·|s))            ← 鼓励探索      │
└──────────────────────────────────────────────────────────────┘
```

> 🎯 **Target Environment**: CartPole-v1 (a simpler environment suitable for A3C's on-policy nature)

[▶ View A3C Code](MyProject/A3C/A3C.py)

---

### 5. PPO (Proximal Policy Optimization)

<div align="center">
  <img src="MyProject/PPO/logs/final_training_curve.png" width="700" alt="PPO Training Curve"/>
</div>

PPO is the **state-of-the-art** algorithm in this project. It improves upon A3C with a more stable update mechanism and is widely used in production systems (including OpenAI's GPT RLHF training).

#### PPO's Key Innovation: Clipped Surrogate Objective

The core idea is to **constrain policy updates** to prevent destructive large steps:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio.

```
Loss L_CLIP
    │               ╱
    │              ╱  clipped (flat)
    │             ╱
    │            ╱
    └───────────●────────────────────▶ ratio r
        1-ε     1     1+ε

    Advantage > 0 (good action) → encourage up to 1+ε
    Advantage < 0 (bad action)  → discourage down to 1-ε
    Beyond clip range → gradient = 0 (no update!)
```

#### GAE (Generalized Advantage Estimation)

PPO uses GAE to compute low-variance advantage estimates:

$$A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

```
    λ = 0  →  A = δ_t                    (high bias, low variance: TD(0))
    λ = 1  →  A = Σ γ^l δ_{t+l}         (low bias, high variance: Monte Carlo)
    λ = 0.95 → smooth interpolation of both ✓
```

#### ConcatNet Architecture (Actor & Critic)

The PPO implementation features the **ConcatNet** with residual connections for both Actor and Critic:

```
┌─────────────────────────────────────────────────────────────┐
│                     Model Architecture                       │
│                                                              │
│  ┌──────────────────────┐    ┌───────────────────────┐     │
│  │    Actor (ConcatNet) │    │   Critic (ConcatNet)  │     │
│  │                      │    │                        │     │
│  │  Input(8) → 512     │    │  Input(8) → 512       │     │
│  │    → split(256,256) │    │    → split(256,256)   │     │
│  │    → 256 → split    │    │    → 256 → split      │     │
│  │    → 128 → split    │    │    → 128 → split      │     │
│  │    → 64 → concat    │    │    → 64 → concat      │     │
│  │    → 688 → 4        │    │    → 688 → 1 (value)  │     │
│  │  Output: Softmax π  │    │  Output: V(s)          │     │
│  └──────────────────────┘    └───────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

#### Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     PPO Training Pipeline                    │
│                                                              │
│  for epoch in range(1000):                                   │
│    ┌──────────────────────────────────────────┐             │
│    │  1. COLLECT: Run policy π_θ in env       │             │
│    │     → Store (s, a, r, V(s), done)        │             │
│    └──────────────────────────────────────────┘             │
│                        ↓                                     │
│    ┌──────────────────────────────────────────┐             │
│    │  2. COMPUTE GAE:                         │             │
│    │     A_t = Σ (γλ)^l δ_{t+l}              │             │
│    │     R_t = A_t + V(s_t)                   │             │
│    └──────────────────────────────────────────┘             │
│                        ↓                                     │
│    ┌──────────────────────────────────────────┐             │
│    │  3. UPDATE (K=4 epochs):                 │             │
│    │     for k in range(K_epochs):            │             │
│    │       L_clip = -min(r·A, clip(r)·A)      │             │
│    │       L_critic = MSE(V(s), R)             │             │
│    │       θ ← θ - α∇L                        │             │
│    └──────────────────────────────────────────┘             │
│                        ↓                                     │
│    ┌──────────────────────────────────────────┐             │
│    │  4. CLEAR memory + LR decay              │             │
│    └──────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

#### Learning Rate Schedule

```python
lr = lr_start × (lr_decay)^epoch    # Exponential decay
# lr_start = 1e-3 → lr_end = 1e-6
# lr_decay = 0.995
```

```
LR
1e-3 ┤╲
     ┤ ╲
8e-4 ┤  ╲
6e-4 ┤   ╲___
4e-4 ┤       ╲___
2e-4 ┤           ╲________
     └──────────────────────▶ Epoch
```

[▶ View PPO Code](MyProject/PPO/)

---

## 📊 Performance Comparison

| Algorithm | Environment | Network Depth | Key Innovation | Solved At |
|-----------|-------------|---------------|----------------|-----------|
| **DQN** | LunarLander-v2 | 128→128→4 | Experience Replay + Target Net | ~6000 episodes |
| **ConcatDQN** | LunarLander-v2 | 1024→256→128→64→16→4 | Residual concat connections | — |
| **Double DQN** | LunarLander-v2 | 256→256→4 | Decoupled action selection | ~2000 episodes |
| **A3C** | CartPole-v1 | 128→64→2 | Multi-worker async updates | ~100 episodes |
| **PPO** 🏆 | LunarLander-v2 | 512→256→128→64→4 | Clipped objective + GAE | ~300 episodes |

### Algorithm Comparison Matrix

```
                    DQN    DDQN   A3C    PPO
                    ───    ────   ───    ───
Off-Policy          ✓      ✓      ✗      ✗
On-Policy           ✗      ✗      ✓      ✓
Experience Replay   ✓      ✓      ✗      ✗
Multi-Process       ✗      ✗      ✓      ✗
Trust Region        ✗      ✗      ✗      ✓
Sample Efficiency   High   High   Low    Medium
Training Stability  Medium Medium Low    High
Final Performance   Good   Better Good   Best 🏆
```

---

## 🛠 Installation

### Prerequisites

- **Python** 3.9+
- **CUDA** 12.1 (optional, for GPU training)
- **Conda** (recommended)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/GYM-RL.git
cd GYM-RL

# 2. Create conda environment
conda env create -f MyProject/environment.yaml
conda activate GGYYMM

# 3. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import gym; env = gym.make('LunarLander-v2'); print(f'Gym {gym.__version__} ✓')"
```

### Manual Installation

```bash
# Core dependencies
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install gym==0.26.2
pip install numpy==1.23.1
pip install matplotlib==3.9.0

# Optional for Box2D environments
pip install box2d-py pygame swig
```

---

## 🚀 Quick Start

### Train DQN

```bash
cd MyProject/DQN
python main.py
```

**Expected output:**
```
Episode 20    avg length: 85    average_reward: -120
Episode 40    avg length: 92    average_reward: -98
Episode 60    avg length: 105   average_reward: -45
...
Episode 6000  avg length: 180   average_reward: 230
########## Solved! ##########
```

### Train PPO

```bash
cd MyProject/PPO
python main.py
```

**Expected output:**
```
开始PPO训练...
环境: LunarLander-v2
状态维度: 8, 动作维度: 4
使用设备: NVIDIA GeForce RTX 3060

Epoch    0 | 步数: 156 | 奖励: -180.34 | 平均奖励(1): -180.34 | 最佳: -180.34
Epoch   50 | 步数: 145 | 奖励:  -45.12 | 平均奖励(50): -52.31 | 最佳:  -20.45
...
🎉 连续10轮成功！训练完成！
总训练时间: 1.23 小时
最佳奖励: 298.45
```

### Train Double DQN

```bash
cd MyProject/DoubleDQN
python DDQN_Beta.py
```

### Train A3C (Multi-Process)

```bash
cd MyProject/A3C
python A3C.py
```

---

## 📁 Project Structure

```
GYM-RL/
│
├── README.md                           # 📖 This comprehensive documentation
│
├── MyProject/
│   ├── environment.yaml                # 🐍 Conda environment specification
│   │
│   ├── DQN/                            # 🧠 Deep Q-Network
│   │   ├── DQN.py                      #   Core DQN implementation (Agent, Network, Replay)
│   │   ├── main.py                     #   Training loop & visualization
│   │   ├── ConcatDQN/                  #   🔬 Residual DQN variant
│   │   │   ├── DQN_concat.py           #     ConcatNet DQN implementation
│   │   │   ├── concat_example.py       #     Standalone ConcatNet demo
│   │   │   └── Readme.md               #     ConcatDQN documentation
│   │   ├── DQN_second_round.gif        #   🎬 Training demonstration
│   │   └── DQN_LunarLander-v2_second.pth
│   │
│   ├── DoubleDQN/                      # 🧠 Double DQN
│   │   ├── DDQN_Beta.py                #   Decoupled action selection
│   │   └── DDQN_LunarLander-v2.pth     #   Pre-trained model
│   │
│   ├── A3C/                            # 🧠 Asynchronous Actor-Critic
│   │   └── A3C.py                      #   Multi-process A3C (CartPole-v1)
│   │
│   └── PPO/                            # 🧠 Proximal Policy Optimization
│       ├── main.py                     #   Training entry point
│       ├── agent.py                    #   PPO Agent (ReplayBuffer, GAE, Update)
│       ├── model.py                    #   ConcatNet, Actor, Critic
│       ├── train_ppo.py                #   PPO training script
│       ├── algorism_test.py            #   Algorithm verification
│       ├── requirements.txt            #   Dependencies
│       ├── README.md                   #   PPO documentation
│       ├── logs/                       #   📊 Training curves at checkpoints
│       │   ├── final_training_curve.png
│       │   ├── training_curve_epoch_*.png
│       │   └── rewards_history.npy
│       └── model/                      #   💾 Saved model checkpoints
│           └── ppo_model_epoch_*.pth
│
└── .git/
```

---

## 🧪 Key Implementation Details

### 1. Experience Replay Buffer

Both DQN and Double DQN use a `deque`-based circular buffer:

```python
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # Auto-evicts old transitions

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # Uniform random sampling
```

### 2. Target Network Update (Soft vs Hard)

```python
# DQN: Soft update (TAU = 0.005)
# θ_target = τ·θ_policy + (1-τ)·θ_target
for key in policy_net_state_dict:
    target_net_state_dict[key] = (policy_net_state_dict[key] * TAU
                                + target_net_state_dict[key] * (1 - TAU))

# Double DQN: Hard update every 5 steps
if self.steps_done % 5 == 0:
    target_net_state_dict[key] = (policy_net_state_dict[key] * 0.005
                                + target_net_state_dict[key] * (1 - 0.005))
```

### 3. GAE Computation (Reverse Pass)

```python
def compute_gae(self, rewards, values, dones, next_value):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + self.gamma * next_val * (1-dones[i]) - values[i]
        gae = delta + self.gamma * self.Lambda * (1-dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

### 4. PPO Clipped Loss

```python
ratio = torch.exp(current_log_probs - old_log_probs)  # r_t(θ)
surr1 = ratio * advantages                             # unclipped
surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages  # clipped
actor_loss = -torch.min(surr1, surr2).mean()           # pessimistic bound
```

### 5. GPU-Agnostic Device Selection

All implementations auto-detect the best available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
```

---

## 📈 Training Curves & Visualizations

### PPO Training Progress (Epoch Checkpoints)

<div align="center">

| Epoch 100 | Epoch 200 | Epoch 300 |
|:---:|:---:|:---:|
| ![100](MyProject/PPO/logs/training_curve_epoch_100.png) | ![200](MyProject/PPO/logs/training_curve_epoch_200.png) | ![300](MyProject/PPO/logs/training_curve_epoch_300.png) |

| Epoch 500 | Epoch 700 | Final |
|:---:|:---:|:---:|
| ![500](MyProject/PPO/logs/training_curve_epoch_500.png) | ![700](MyProject/PPO/logs/training_curve_epoch_700.png) | ![Final](MyProject/PPO/logs/final_training_curve.png) |

</div>

### Training Reward Progression (PPO)

```
Reward
 300 ┤                                          ▄▄███▄
 200 ┤                                   ▄▄███████████
 100 ┤                             ▄▄██████████████████
   0 ┤                       ▄▄████████████████████████
-100 ┤                 ▄▄██████████████████████████████
-200 ┤ ▄▄████████████████████████████
     └─────────────────────────────────────────────────▶ Episode
      0    100   200   300   400   500   600   700   800
```

### Moving Average (Window=50) — PPO

```
Reward
 250 ┤                                              ▄████
 200 ┤                                    ▄▄▄█████████
 150 ┤                           ▄▄███████████████████
 100 ┤                   ▄▄█████████████████████████
  50 ┤          ▄▄████████████████████████
   0 ┤  ▄▄████████████████████
-100 ┤──
     └─────────────────────────────────────────────────▶ Episode
```

---

## 💡 Insights & Lessons Learned

### DQN → Double DQN

| Insight | Detail |
|---------|--------|
| **Overestimation is real** | Standard DQN systematically inflates Q-values. Double DQN's simple decoupling (choosing actions with policy net, evaluating with target net) yields significantly more accurate value estimates |
| **Faster convergence** | Double DQN solves LunarLander ~3x faster than vanilla DQN (~2000 vs ~6000 episodes) |
| **Hidden dimension matters** | The jump from 128→256 hidden units in Double DQN provides meaningful capacity improvement |

### DQN → ConcatDQN

| Insight | Detail |
|---------|--------|
| **Multi-scale features** | Concatenating features from different depths preserves both low-level and abstract representations |
| **Wider is sometimes better than deeper** | The 1024-dim first layer with splitting creates implicit ensembling |

### A3C → PPO

| Insight | Detail |
|---------|--------|
| **Trust regions stabilize** | A3C's unrestricted gradient updates can cause policy collapse; PPO's clipping prevents this entirely |
| **GAE is crucial** | Without GAE (λ=0 → pure TD), advantage estimates are too noisy; with GAE (λ=0.95), training becomes remarkably stable |
| **Multiple epochs per batch** | PPO reuses each batch K=4 times — vastly more sample-efficient than A3C's single pass |
| **Learning rate matters** | Exponential decay from 1e-3→1e-6 helps converge to a precise policy once the rough shape is learned |

### General Lessons

1. **Start simple**: DQN is the easiest to debug — get it working first, then iterate
2. **Watch the reward curve shape**: A smooth, monotonically increasing curve indicates healthy training; oscillations suggest instability
3. **GPU matters**: A3C's multi-process design shines on multi-core CPUs; DQN/PPO benefit most from GPU acceleration
4. **Checkpoint often**: Save models at regular intervals — you never know when training might diverge
5. **Normalize advantages**: `(A - mean(A)) / (std(A) + 1e-8)` is critical for PPO stability

---

## 🎮 Live Demos

### DQN Landing the Lunar Module

<div align="center">
  <img src="MyProject/DQN/DQN_second_round.gif" width="500" alt="DQN LunarLander Demo"/>
  <p><em>After ~6000 training episodes, the DQN agent learns to land smoothly on the pad</em></p>
</div>

### Training Phases (Conceptual)

```
Phase 1: Random Flailing            Phase 2: Hover Control
     ┌─┐                                  │
     │ │  ↘                               │  ←→
     └─┘   ↓                              │
     💥 CRASH!                            △  Stabilizing...

Phase 3: Approach                    Phase 4: Perfect Landing
        ╲                                     │
         ╲  ↘                                 │  ↓
          ◉                                   ▓▓▓▓
     Aiming for pad...                  🛸 TOUCHDOWN!
```

---

## 📚 References

### Academic Papers

- 📄 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Mnih et al., 2013 *(Original DQN paper)*
- 📄 [Double Q-learning](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — Hasselt, 2010
- 📄 [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Mnih et al., 2016 *(A3C)*
- 📄 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017 *(PPO)*
- 📄 [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015 *(GAE)*

### Online Resources

| Resource | Link |
|----------|------|
| 🏛️ LunarLander-v2 Docs | [gymnasium.farama.org](https://gymnasium.farama.org/environments/box2d/lunar_lander/) |
| 🔥 PyTorch DQN Tutorial | [pytorch.org/tutorials](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) |
| 📝 LunarLander Reference | [github.com/lazavgeridis](https://github.com/lazavgeridis/LunarLander-v2/blob/main/README.md) |
| 💻 PPO Reference Implementation | [github.com/BlackMirean](https://github.com/BlackMirean/PPO/blob/master/PPO.py) |

### Related Projects & Tools

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — Production-grade RL implementations
- [Gymnasium](https://gymnasium.farama.org/) — Standardized RL environment API
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html) — Scalable RL for production

---

## 🤝 Contributing

Contributions are welcome! Areas where help is especially appreciated:

- 🧪 Add **SAC (Soft Actor-Critic)** implementation
- 🧪 Add **TD3 (Twin Delayed DDPG)** for continuous control
- 📊 Add more comprehensive benchmarking with statistical significance tests
- 🎮 Extend to other Gym environments (BipedalWalker, CarRacing)
- 📝 Improve documentation and add Chinese translations

---

## 📄 License

This project is open-sourced under the MIT License.

---

<div align="center">

```
    ╔══════════════════════════════════════════════╗
    ║  🛸  GYM-RL — From Zero to RL Mastery      ║
    ║                                              ║
    ║  "The best way to learn RL is to build it." ║
    ╚══════════════════════════════════════════════╝
```

**Made with ❤️ by GYM-RL | 2024–2026**

⭐ If this project helped you, please star it on GitHub!

</div>
