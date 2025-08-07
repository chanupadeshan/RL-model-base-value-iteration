import numpy as np
import gymnasium as gym
import time




# Extract MDP model
def extract_model(env):
    #ns : total number of states
    #na : total number of actions
    ns, na = env.observation_space.n, env.action_space.n
    print(f"Number of states: {ns}, Number of actions: {na}")
    P = np.zeros((ns, na, ns))  # P: transition probabilities
    R = np.zeros((ns, na))  # R: rewards


    for s in range(ns):
        for a in range(na):
            # Get the transition probabilities and rewards from the env
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                # fill the transition probabilites to P
                P[s, a, next_state] = prob
                # Fill the rewards to R
                R[s, a] = reward
    return P, R, ns, na




# gamma : discount factor
# theta : tiny threshold to stop the value iteration
def value_iteration(P,R,gamma=0.99,theta=1e-6):
    ns,na,_ = P.shape
    # initialize the V array to store the state values for each state
    V=np.zeros(ns)

    while True:
        delta = 0
        # Loop through each state to update its state value
        for s in range(ns):
            v = V[s]
            ## calculate the maximum state value for the current state
            """
            for a in range(na): It goes over all actions in the current state
            for s2 in range(ns): It goes over all next states
            R[s, a] + gamma * V[s2] : this is calculate total rewards (immediate reward + discounted future reward)
            sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(ns)) : this calculates the state value for the current state and current(single) action
            """
            V[s] = max(sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(ns))for a in range(na))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break


    # select the best action values for each state
    policy = np.zeros((ns, na))
    for s in range(ns):
        # Calculate the all action values for the current state
        Qs = [sum(P[s, a, s2] * (R[s, a] + gamma * V[s2]) for s2 in range(ns)) for a in range(na)]
        # Select the best action value for the current state
        best_action = np.argmax(Qs)
        #update the 1 as the best action for the current state (eg:0,0,1,0)
        policy[s, best_action] = 1
    return policy, V
    



def run_policy(env,policy,delay=0.9):
    state,_ = env.reset() # Start a new episode
    done = False # done is True when the episode is finished
    total_reward = 0
    na = env.action_space.n

    while not done:
        action = np.argmax(policy[state])
        # state : next state
        # reward : immediate reward
        # terminated : the episode ended normally (goal reached, failure happened, etc.)
        # truncated : the episode ended due to a time limit
        state,reward,terminated,truncated,_ = env.step(action)

        print(f"State: {state}, Action: {action}, Reward: {reward}")
        total_reward += reward
        time.sleep(delay)

        done = terminated or truncated

    env.close()
    print(f"Total reward: {total_reward}")





if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, render_mode="human", is_slippery=False)
    P, R, ns, na = extract_model(env)
    policy, V = value_iteration(P,R)

    print("Optimal State value function:")
    print(V)

    run_policy(env, policy)