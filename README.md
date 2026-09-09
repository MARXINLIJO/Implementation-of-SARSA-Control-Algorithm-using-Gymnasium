# Implementation of SARSA Control Algorithm using Gymnasium

## Aim

To implement the SARSA control algorithm using the Gymnasium FrozenLake-v1 environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

## Problem Statement

To develop a reinforcement learning agent using the SARSA (State-Action-Reward-State-Action) algorithm. The agent must learn the best actions in a custom 4×4 FrozenLake environment, starting from state 12 and reaching the goal state 3 while avoiding the hole states.

## Software Requirements

* Python 3.x
* Google Colab / Jupyter Notebook
* Gymnasium
* NumPy
* Matplotlib

## Environment Description

A custom 4×4 FrozenLake environment is used.

### Custom Map

```text
F F F G
F H F H
F F F F
S F F F
```

### State Representation

```text
0   1   2   3
4   5   6   7
8   9   10  11
12  13  14  15
```

* **Start state:** 12
* **Goal state:** 3
* **Hole states:** 5, 7, 11
* **F:** Frozen surface
* **H:** Hole
* **S:** Start
* **G:** Goal

The environment uses `is_slippery=False`, so the agent moves exactly in the direction selected.

## Actions

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

## Theory

SARSA stands for:

```text
Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁, Aₜ₊₁
```

SARSA is an **on-policy Temporal Difference reinforcement learning algorithm**. It updates the Q-value using the action actually selected in the next state.

### SARSA Update Rule

```text
Q(Sₜ, Aₜ) ← Q(Sₜ, Aₜ) +
α [Rₜ₊₁ + γQ(Sₜ₊₁, Aₜ₊₁) − Q(Sₜ, Aₜ)]
```

### Symbols

| Symbol   | Meaning                                       |
| -------- | --------------------------------------------- |
| `Sₜ`     | Current state                                 |
| `Aₜ`     | Current action                                |
| `Rₜ₊₁`   | Reward received after taking action `Aₜ`      |
| `Sₜ₊₁`   | Next state                                    |
| `Aₜ₊₁`   | Next action selected using the current policy |
| `α`      | Learning rate                                 |
| `γ`      | Discount factor                               |
| `Q(s,a)` | Action-value function                         |

## Parameters Used

| Parameter                 |  Value |
| ------------------------- | -----: |
| Number of episodes        | 50,000 |
| Maximum steps per episode |    100 |
| Learning rate (`α`)       |    0.1 |
| Discount factor (`γ`)     |   0.99 |
| Initial epsilon           |    1.0 |
| Minimum epsilon           |   0.05 |
| Epsilon decay             | 0.9999 |

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

* With probability `ε`, the agent **explores** by selecting a random action.
* With probability `1 − ε`, the agent **exploits** by selecting the action with the highest Q-value.

```text
a =
{
    random action,              with probability ε

    argmax Q(s,a),               with probability 1 − ε
}
```

Initially, `epsilon = 1.0`, so the agent performs more exploration.

After every episode, epsilon is reduced using:

```python
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

This gradually changes the agent from exploration to exploitation.

## Algorithm

1. Create the custom 4×4 FrozenLake environment.
2. Set the start state as 12 and the goal state as 3.
3. Initialize the Q-table with zeros.
4. Set the learning parameters `α`, `γ`, and `ε`.
5. Reset the environment at the beginning of each episode.
6. Select an initial action using the epsilon-greedy policy.
7. Execute the selected action.
8. Observe the next state and reward.
9. Select the next action using the epsilon-greedy policy.
10. Update the Q-value using the SARSA equation.
11. Move to the next state and next action.
12. Repeat until the episode terminates.
13. Reduce epsilon after every episode.
14. Store the reward obtained in each episode.
15. Calculate the state-value function from the learned Q-table.
16. Generate the learned policy using the maximum Q-value.
17. Calculate the average reward over the last 1000 episodes.
18. Plot the learning curve.

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

custom_map = [
    "FFFG",
    "FHFH",
    "FFFF",
    "SFFF"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False
)

start_state = 12
goal_state = 3

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 50000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9999

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, _ = env.reset()

    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:

            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

            break

        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # SARSA update
        Q[state, action] += alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        state = next_state
        action = next_action

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    episode_rewards.append(total_reward)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

print("\nStart State:", start_state)
print("Goal State:", goal_state)

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)

print("Name: Marxin Lijo ")
print("Reg no: 212223240085")

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title("SARSA Learning Curve - FrozenLake")

plt.grid(True)
plt.show()

env.close()
```

## Output
<img width="1148" height="1370" alt="image" src="https://github.com/user-attachments/assets/20ae6df8-5dda-400d-a7bd-c312e3e91ea0" />

<img width="667" height="426" alt="image" src="https://github.com/user-attachments/assets/c1653dcb-2d1e-48e0-a1bd-4b48867e2399" />



## Possible Successful Path

A possible successful path learned by the agent is:

```text
12 → 13 → 14 → 10 → 6 → 2 → 3
```

### Corresponding Actions

```text
Right → Right → Up → Up → Up → Right
```

## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake environment. The agent learned Q-values through repeated interaction with the environment and gradually learned a policy to move from start state 12 to goal state 3 while avoiding the holes.

## Inference

The experiment demonstrates that SARSA can learn an effective policy through trial and error. Initially, the agent explores the environment using a high epsilon value. As epsilon decreases, the agent increasingly selects actions with higher Q-values.

The Q-table stores the expected value of each state-action pair, while the learned policy selects the action with the maximum Q-value. The learning curve shows how the agent's performance changes during training, demonstrating the learning process of the SARSA algorithm.
