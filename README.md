<img width="1398" height="138" alt="image" src="https://github.com/user-attachments/assets/fdbdb7d8-67dc-4fb9-97ea-c732830a9ed4" /># EX 5. Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

```python
# Write your code here
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# 1. Create Environment
# -------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of States :", n_states)
print("Number of Actions:", n_actions)


# -------------------------------------------------
# 2. Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# 3. Parameters
# -------------------------------------------------

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

num_episodes = 10000

# Store reward of every episode
episode_rewards = []


# -------------------------------------------------
# 4. Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    else:
        return np.argmax(Q[state])


# -------------------------------------------------
# 5. Monte Carlo Control
# -------------------------------------------------

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Store episode as:
    # (state, action, reward)
    episode_data = []

    total_reward = 0

    terminated = False
    truncated = False


    # -------------------------------------------------
    # Generate Complete Episode
    # -------------------------------------------------

    while not terminated and not truncated:

        # Select action using epsilon-greedy
        action = epsilon_greedy_action(
            state,
            epsilon
        )

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Store experience
        episode_data.append(
            (state, action, reward)
        )

        # Add reward
        total_reward += reward

        # Move to next state
        state = next_state


    # -------------------------------------------------
    # Calculate Return and Update Q-table
    # -------------------------------------------------

    G = 0

    # Traverse episode backwards
    for state, action, reward in reversed(episode_data):

        # Calculate return
        G = reward + gamma * G

        # Incremental Monte Carlo update
        Q[state, action] = (
            Q[state, action]
            + alpha * (G - Q[state, action])
        )


    # -------------------------------------------------
    # Store Episode Reward
    # -------------------------------------------------

    episode_rewards.append(total_reward)


    # -------------------------------------------------
    # Decay Epsilon
    # -------------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# ============================================================
# 6. Calculate State-Value Function
# ============================================================

state_values = np.max(Q, axis=1)


# ============================================================
# 7. Calculate Optimal Policy
# ============================================================

optimal_policy = np.argmax(Q, axis=1)


# ============================================================
# 8. Display Results
# ============================================================


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nName: Vignesh M")
    print("Register Number: 212223240176")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


# -------------------------------------------------
# Final Q-table
# -------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(
        Q,
        3
    )
)


# -------------------------------------------------
# State-Value Function
# -------------------------------------------------

print_value_function(
    state_values
)


# -------------------------------------------------
# Learned Policy
# -------------------------------------------------

print_policy(
    optimal_policy
)


# -------------------------------------------------
# Average Reward
# -------------------------------------------------

success_rate = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    round(success_rate, 3)
)


# ============================================================
# 9. Learning Curve
# ============================================================

window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)


plt.figure(figsize=(10, 5))

plt.plot(
    moving_average
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "On-Policy Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.show()


# ============================================================
# 10. Close Environment
# ============================================================

env.close()



```

---

## Output

Final Q-table:
<img width="1253" height="372" alt="image" src="https://github.com/user-attachments/assets/b3fa4416-5e39-47c3-aa24-b449f4864cbf" />



Estimated State-Value Function:
<img width="1250" height="130" alt="image" src="https://github.com/user-attachments/assets/d9c94197-18f2-4d1c-a4b6-f24746deac78" />




Learned Policy:
<img width="1317" height="206" alt="image" src="https://github.com/user-attachments/assets/466d4c5a-8454-470f-af76-e887ebeca9b6" />





Average reward over last 1000 episodes: 
<img width="1553" height="559" alt="image" src="https://github.com/user-attachments/assets/a584e3b9-4514-4c2c-a35e-755d83dea4da" />


Final Q-table:
<img width="1190" height="421" alt="image" src="https://github.com/user-attachments/assets/9f788d86-a1d2-4d00-8b9f-9d211ecf927f" />



Estimated State-Value Function:
<img width="1398" height="138" alt="image" src="https://github.com/user-attachments/assets/47ecf4bd-7223-43a4-87d4-b657ddf979c0" />






Learned Policy:
<img width="738" height="203" alt="image" src="https://github.com/user-attachments/assets/d35e1119-d22a-43f8-a238-5711f1656420" />


Average reward over last 5000 episodes: 
<img width="1616" height="553" alt="image" src="https://github.com/user-attachments/assets/dfa3878e-3706-45e4-bef8-7bb5ab0a93d9" />


---

## Result
```text
The agent successfully learned a policy for FrozenLake using On-Policy Monte Carlo Control.


```
---

## Inference
```text
Repeated episodes improve the Q-values, allowing the agent to select better actions and reach the goal more effectively.


```





---

