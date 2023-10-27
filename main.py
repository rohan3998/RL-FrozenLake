import numpy as np
#import tools
import gym
import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v1')

print(env.action_space)
print(env.observation_space.n)
num_states = env.observation_space.n
num_actions = env.action_space.n
#print(env.P[0])


def value_iteration(env, gamma = 0.9, theta = 1e-4):

    values = []
    V = np.zeros(num_states)
    values.append(V.copy())
    while(True):
        diff = 0
        for s in range(num_states):
            v_curr = V[s]
            max_sum_action = float('-inf')
            for a in range(num_actions):

                sum_action = 0
                for p,s_,r,done in env.P[s][a]:
                    sum_action += p*(r + gamma*V[s_])
                max_sum_action = max(max_sum_action,sum_action)

            V[s] = max_sum_action
            diff = max(diff,abs(v_curr - V[s]))
        values.append(V.copy())
        #print(V)
        if diff < theta:
            break
        #print(diff)
    norms = []

    for i in range(len(values)-1):
        norms.append(np.linalg.norm(values[i+1] - values[i]))

    indices = range(len(norms))
    #print(values)
    plt.plot(indices, norms, marker = '.')
    plt.title("Value function diff vs. k")
    plt.xlabel("k")
    plt.ylabel("Value function diff")
    plt.show()
    policy = np.zeros(num_states)
    #print(V)

    for s in range(num_states):
        Q = np.zeros(num_actions)

        for a in range(num_actions):
            sum_action = 0
            for p,s_,r,done in env.P[s][a]:
                sum_action += p*(r + gamma*V[s_])
            Q[a] = sum_action
        
        policy[s] = np.argmax(Q)

    print(policy)

    plt.plot(range(num_states), V, marker = '.')
    plt.title("Optimal Value function")
    plt.xlabel("State")
    plt.ylabel("Value function")

    plt.show()

    plt.plot(range(num_states), policy, marker = '.')
    plt.title("Optimal Policy")
    plt.xlabel("State")
    plt.ylabel("Action")

    plt.show()
    return V, policy


def policy_evaluation(policy, env, gamma, theta):
    V = np.zeros(env.observation_space.n)
    while(True):
        diff = 0
        for s in range(num_states):
            v_curr = V[s]
            a = policy[s]
            sum_action = 0
            for p,s_,r,done in env.P[s][a]:
                sum_action += p*(r + gamma*V[s_])

            V[s] = sum_action
            diff = max(diff,abs(v_curr - V[s]))
        if diff < theta:
            break
    return V

def policy_improvement(value_function, env, gamma):

    policy = np.zeros(env.observation_space.n)
    for s in range(num_states):
        Q = np.zeros(num_actions)
        for a in range(num_actions):
            sum_action = 0
            for p,s_,r,done in env.P[s][a]:
                sum_action += p*(r + gamma*value_function[s_])
            Q[a] = sum_action 
        policy[s] = np.argmax(Q)

    return policy

def policy_iteration(env, gamma = 0.9, theta = 1e-7, max_iter = 1000):
    policy = np.zeros(env.observation_space.n)
    value_functions = []
    for i in range(max_iter):
        old_policy = policy.copy()
        value_function = policy_evaluation(policy, env, gamma, theta)
        policy = policy_improvement(value_function, env, gamma)
        value_functions.append(value_function.copy())
        if (old_policy == policy).all():
            break

    norms = []

    for i in range(len(value_functions)-1):
        norms.append(np.linalg.norm(value_functions[i+1] - value_functions[i]))

    indices = range(len(norms))
    #print(values)
    plt.plot(indices, norms, marker = '.')
    plt.title("V(pi_k) - V(pi_(k-1)) vs. k")
    plt.xlabel("policy")
    plt.ylabel("V(pi_k) - V(pi_(k-1)")
    plt.show()
    #print(policy)
    return policy, value_function

value_iteration(env)
policy, value_function = policy_iteration(env)
print(policy)