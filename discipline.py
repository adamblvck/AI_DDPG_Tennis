import numpy as np
import wandb
from collections import namedtuple, deque
import torch

def save_agent(agents, checkpoint_name):
    for i, agent in enumerate(agents):
        torch.save(agent.actor_local.state_dict(), "{}_{}_actor.pth".format(checkpoint_name,i))
        torch.save(agent.critic_local.state_dict(), "{}_{}_critic.pth".format(checkpoint_name,i))

def load_agent(agents, checkpoint_name):
    for i, agent, in enumerate(agents):
        agent.actor_local.load_state_dict(torch.load("{}_{}_actor.pth".format(checkpoint_name,i)))
        agent.critic_local.load_state_dict(torch.load("{}_{}_critic.pth".format(checkpoint_name,i)))

def env_agent_reset(env, agents, state_size_per_agent, num_agents, brain_name, train=True):
    # combine the states of both agents into a single state (1D)
    s = env.reset(train_mode=train)[brain_name].vector_observations.reshape((1,
                                                                             num_agents*state_size_per_agent))
    
    # reset agents
    for agent in agents:
        agent.reset()
        
    return s, np.zeros(num_agents)
    
def env_manifest_action(env, actions, brain_name, num_agents, state_size_per_agent):
    """ Takes a step in the environment, returns r, s_, complete the (s,a,r,s_) tuple """
    reaction = env.step(actions)[brain_name]
    
    # extract required values
    r = reaction.rewards
    s_ = reaction.vector_observations.reshape((1, num_agents*state_size_per_agent))
    dones = reaction.local_done
    
    return r, s_, dones

def log(i_episode, score, scores_window, solved=False):
    
    max_scores = np.max(score)
    mean_scores_window = np.mean(scores_window)
    
    # log avg score for this max_scores across agents
    wandb.log({"mean_score": max_scores})
    wandb.log({"mean_window_score": mean_scores_window})
    
    if not solved and i_episode%50 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}\tMax score: {:.2f}'.format(i_episode, 
                                                                          mean_scores_window, 
                                                                          max_scores, end=""))
    elif solved:
        print('\nEnvironment solved in {} episodes!\tAverage score: {:.2f}'.format(i_episode,
                                                                                     mean_scores_window))
        
def agents_think(agents, s):
    
    actions = [agent.think(s, True) for agent in agents] # let the agents think
    return np.hstack(tuple(actions))  # stack actions
    
def agents_accept_reaction(agents, t, s, a, r, s_, done):
    for i, agent in enumerate(agents):
        agent.accept(t, s, a, r[i], s_, done[i], i) # 0 because it's agent 0
    
def train_ddpg_tennis(env, agents, num_agents, brain_name, n_episodes=10000, max_t=1000, checkpoint_name='checkpoint', keep_training=False):
    scores = []
    scores_window = deque(maxlen=100)
    
    # size of the state space shared by the agents
    state_size_per_agent = 24
    t = 0
    
    for i_episode in range(1, n_episodes+1):
        
        # reset => env, agent, score
        s, score = env_agent_reset(env, agents, state_size_per_agent, num_agents, brain_name)
        
        # Run for as long as "done" is not reached
        # - Agent thinks of an action - agents_think()
        # - Environement accepts action - env_manifest_action()
        # - Agent accepts his reality - agents_accept_reaction()
        while True:
            
            # Mr AI, what action should we take?
            a = agents_think(agents, s)
            
            # Take the action in the environment
            r, s_, dones = env_manifest_action(env, a, brain_name, num_agents, state_size_per_agent)
            
            # Mr AI, if you choose to accept
            agents_accept_reaction(agents, t, s, a, r, s_, dones)
            
            # prep next step & count score
            s = s_
            score += r
            t+=1
            
            if np.any(dones): # if any agent is at it's completion
                break
                
        # store MAX scores
        scores.append(np.max(score))
        scores_window.append(np.max(score))
        
        # log scores
        log(i_episode, score, scores_window)
        
        # if i_episode % 1000 == 0 :
        #    save_agent(agent, checkpoint_name+'_'+str(i_episode))
        
        if i_episode % 50 == 0 and np.mean(scores_window) >= 0.5: # log avg score in last 100 episodes
            log(i_episode, score, scores_window, solved=True)
            save_agent(agents, checkpoint_name+'_'+str(i_episode))
            if not keep_training:
                break
        
    return scores

def play_ddpg_tennis(env, agent_0, agent_1, num_agents, brain_name, n_episodes=6):
    scores = []
    scores_window = deque(maxlen=100)
    max_score = np.Inf
    
    # size of the state space shared by the agents
    state_size_per_agent = 24
    
    # put into play mode
    _ = env.reset(train_mode=False)[brain_name]
    
    for i_episode in range(1, n_episodes):
        
        # reset => env, agent, score
        s, score = env_agent_reset(env, [agent_0, agent_1], state_size_per_agent, num_agents, \
                                                               brain_name, train=False)

        # run for max_t time (or when `done` is reached)
        # - Agent thinks of an action - agent.think
        # - Environement accepts action - env.step
        # - Agent accepts his reality - agent.accept
        while True:

            # Mr AI, what action should we take?
            a = agents_think([agent_0, agent_1], s)

            # Take the action in the environment
            r, s_, dones = env_manifest_action(env, a, brain_name, num_agents, state_size_per_agent)

            # Mr AI, if you choose to accept
            # agents_accept_reaction([agent_0, agent_1], t, s, a, r, s_, dones)

            # prep next step & count score
            s = s_
            score += r

            if np.any(dones): # if any agent is at it's completion
                break

        # store MAX scores
        scores.append(np.max(score))
        scores_window.append(np.max(score))

        # log scores
        # log(i_episode, score, scores_window)
        print (scores_window)

        # if i_episode % 1000 == 0 :
        #    save_agent(agent, checkpoint_name+'_'+str(i_episode))

        if np.mean(scores_window) >= 0.5: # log avg score in last 100 episodes
            #log(i_episode, score, scores_window, solved=True)
            #save_agent(agent, checkpoint_name+'_'+str(i_episode))
            break
