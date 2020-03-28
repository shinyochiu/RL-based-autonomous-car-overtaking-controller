import numpy as np
import random
from maddpg import AgentTrainer

def maxQ_act(obs, act_space_num, q_debug):
    if len(obs) == 16:
        obs = np.transpose(np.expand_dims(obs, axis=-1))
    else:
        obs = np.squeeze(obs)
    q_value = np.zeros((obs.shape[0], act_space_num))
    for act in range(act_space_num):
        temp_act = np.zeros((obs.shape[0], act_space_num))
        if obs.shape[0] > 1:
            temp_act[:][act] = 1
            q_values = q_debug['q_values'](*([obs] + [temp_act]))
            for q in range(len(q_value)):
                q_value[q][act] = q_values[q]
        else:
            temp_act[0][act] = 1
            q_value[0][act] = q_debug['q_values'](*([obs] + [temp_act]))
    q_value = np.argmax(q_value, axis=1)
    policy_act = np.zeros((obs.shape[0], act_space_num))
    for q in range(len(q_value)):
        policy_act[q][q_value[q]] = 1

    return policy_act if obs.shape[0] > 1 else np.squeeze(policy_act)

class Fuzzy_control(AgentTrainer):
    def __init__(self, model, obs_shape_n, act_space_n, agent_num, args, flag=0):
        # only 1 agent's observations contained
        self.n = len(obs_shape_n)
        self.act_space = act_space_n
        self.args = args

    def action(self, obs, flag):
        """

        :param obs:(dis2goal + ang_err + vel + omg + entity_dis + entity_ang)
        :param flag:
        :return:
        """
        gamma = 0.5
        act = np.zeros(3)
        d_safe = 15*obs[2]
        road_width = 0.33
        if flag == 0 and obs[4] > d_safe:
            act[2] = 1
            return act, flag
        else:
            flag = 1

        if flag == 1:
            if obs[4]*np.sin(obs[10]) < road_width:
                act[1] = act[2] = 1
            elif obs[4]*np.sin(obs[10])-1.5*road_width < 0.01 and flag == 1:
                w = 0-obs[3]
                if w > 0:
                    if w > 0.5:
                        act[1] = act[2] = 1
                    else:
                        act[2] = 1
                        act[1] = w*2
                elif w < 0:
                    if w < -0.5:
                        act[0] = act[2] = 1
                    else:
                        act[2] = 1
                        act[0] = w * 2
                if obs[4]*np.sin(obs[10])-2*road_width < 0.01:
                    flag = 2
            else:
                act[0] = act[2] = 1
            return act, flag

        if flag == 2:
            act[2] = 1
            if obs[4] > d_safe:
                flag = 3
            return act, flag

        if flag == 3:
            if obs[4]*np.sin(obs[10]) > road_width:
                act[2] = np.max(0.2,np.min(0.5, gamma*obs[4]))
                act[0] = 1
            elif obs[4] * np.sin(obs[10]) - 0.5 * road_width < 0.01:
                w = 0 - obs[3]
                if w > 0:
                    if w > 0.5:
                        act[2] = np.max(0.2, np.min(0.5, gamma * obs[4]))
                        act[1] = 1
                    else:
                        act[2] = np.max(0.2, np.min(0.5, gamma * obs[4]))
                        act[1] = w * 2
                elif w < 0:
                    if w < -0.5:
                        act[2] = np.max(0.2, np.min(0.5, gamma * obs[4]))
                        act[0] = 1
                    else:
                        act[2] = np.max(0.2, np.min(0.5, gamma * obs[4]))
                        act[0] = w * 2
                if obs[4] * np.sin(obs[10]) < 0.01:
                    flag = 4
            else:
                act[2] = np.max(0.2, np.min(0.5, gamma * obs[4]))
                act[1] = 1
            return act, flag
        if flag == 4:
            act[2] = 0.5 - gamma * obs[4]
            return act, flag









