# RL Model-Based Value Iteration

This project implements a model-based reinforcement learning agent that uses value iteration to find the optimal policy for the FrozenLake-v1 environment from the Gymnasium library.

## How it works

The agent first builds a model of the environment, consisting of the transition probabilities and reward function. Then, it uses value iteration to find the optimal policy.

### Selecting the Best Policy

The best policy is determined by using both state values and action values.

1. **State Value Calculation**: The value iteration algorithm is used to compute the optimal value function, `V(s)`, for each state `s`. This represents the maximum expected return starting from state `s`.

2.  **Action Value Calculation**: Once the optimal state values are found, we can determine the best policy. For each state `s`, we calculate the value of taking each action `a`. This is the action-value function, `Q(s, a)`, calculated as follows:

    `Q(s, a) = R(s, a) + γ * Σ [P(s'|s, a) * V(s')]`

    Where:
    - `R(s, a)` is the immediate reward for taking action `a` in state `s`.
    - `γ` is the discount factor.
    - `P(s'|s, a)` is the probability of transitioning to state `s'` from state `s` after taking action `a`.
    - `V(s')` is the optimal value of the next state `s'`.

3.  **Best Policy Selection**: The optimal policy `π*(s)` for each state `s` is to select the action `a` that maximizes the action value `Q(s, a)`.

    `π*(s) = argmax_a Q(s, a)`

This project demonstrates this process by first running value iteration to get `V(s)` and then using `V(s)` to compute `Q(s,a)` and select the best action for each state.
