import time
from collections import deque

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch

from maddpg import MADDPGAgent


def run_episode(env, brain_name, agents, n_episodes=2000, max_steps=1000, update_every=2, update_count=4):
    score_total_means = deque(maxlen=100)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        # Reset Env and Agent
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        
        # total score of 20 agents
        total_rewards = np.zeros(len(agents))
        
        for agent in agents:
            agent.reset()
        
        start_time = time.time()
        
        for t in range(max_steps):
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            env_info = env.step(actions)[brain_name]
            
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done 
            
            for i, agent in enumerate(agents):
                agent.step(states[i], actions[i], actions[1-i], rewards[i], next_states[i], dones[i])
            
            states = next_states
            total_rewards += env_info.rewards
            
            if t % update_every == 0:
                for _ in range(update_count):
                    for agent in agents:
                        agent.start_learn()

            if np.any(dones):
                break

    
        duration = time.time() - start_time
        
        min_score = np.min(total_rewards)
        max_score = np.max(total_rewards)
        scores.append(max_score)
        score_total_means.append(max_score)
        total_average = np.mean(score_total_means)
        
        print('\rEpisode {}({} steps)\tTotal Average Score: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'
              .format(i_episode, t, total_average, min_score, max_score, duration))

        if i_episode % 10 == 0:
            for agent in agents:
                agent.save()
            
        if total_average >= 0.5 and i_episode >= 100:
            print('Problem Solved after {} epsisodes!! Total Average score: {:.2f}'.format(i_episode, total_average))
            for agent in agents:
                agent.save("solved")
            break
    
    return scores

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    agents = [MADDPGAgent(i, state_size, action_size, num_agents, random_seed=7, device=device) for i in range(num_agents)]
    scores = run_episode(env, brain_name, agents)
    plot_scores(scores)

if __name__ == "__main__":
    main()