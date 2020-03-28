import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Static_obs, Dynamic_obs
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        #world.damping = 1

        num_good_agents = 1
        num_adversaries = 0
        num_agents = num_adversaries + num_good_agents
        num_static_obs = 4
        num_landmarks = 8
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = False#True if i == 0 else False
            agent.silent = False #if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.1*np.sqrt(2)
            agent.accel = 0.05 if agent.adversary else 3
            agent.max_speed = 1
            agent.u_noise = 0.3
            agent.crash = 0  # calculate how many time the agent crashed
        world.static_obs = [Static_obs() for i in range(num_static_obs)]
        for i, static_obs in enumerate(world.static_obs):
            static_obs.name = 'static_obs %d' % i
            static_obs.collide = True
            static_obs.movable = False
            static_obs.boundary = False
            static_obs.size = 1.5*np.sqrt(2)
            static_obs.accel = 0.05
            static_obs.max_speed = 0.1
            static_obs.color = np.array([0.8, 0.8, 0.8])
            static_obs.state.p_vel = np.zeros(world.dim_p)
            static_obs.state.p_ang = 0.0
            static_obs.state.p_pos = np.array(([
                2.5 * np.sqrt(2) * np.cos(np.pi / 4 + 2 * np.pi * i / len(world.static_obs)),
                2.5 * np.sqrt(2) * np.sin(np.pi / 4 + 2 * np.pi * i / len(world.static_obs))]))
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False
            landmark.size = 0.1 * np.sqrt(2)
            landmark.color = np.array([1, 1, 0.4])
            if i < 4:
                landmark.state.p_pos = np.array([0, -3+2*i])
                landmark.points = [(-0.01,-0.55),(-0.01,0.55),(0.01,0.55),(0.01,-0.55)]
            else:
                landmark.state.p_pos = np.array([-3+2*(i-4), 0])
                landmark.points = [(-0.55, -0.01), (-0.55, 0.01), (0.55, 0.01), (0.55, -0.01)]
        # make initial conditions
        self.reset_world(world)
        return world

    def set_path(self, path_type, start):
        if path_type == 'square':
            xs = np.array([-start, start, start, -start, -start])
            ys = np.array([-start, -start, start, start, -start])
            xys = list(zip(xs, ys))
        elif path_type == 'line':
            xs = np.array([start[0], 4])
            ys = np.array([start[1], start[1]])

            xys = list(zip(xs, ys))
        return xys

    '''def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 0.5 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)


        return boundary_list'''


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            agent.crash = 0  # calculate how many time the agent crashed
        # random properties for landmarks
        num_dynamic_obs = 1#np.random.randint(1,3)
        world.dynamic_obs = [Dynamic_obs() for i in range(num_dynamic_obs)]

        for i, dynamic_obs in enumerate(world.dynamic_obs):
            dynamic_obs.name = 'dynamic_obs %d' % i
            dynamic_obs.collide = True
            dynamic_obs.movable = True
            dynamic_obs.boundary = False
            dynamic_obs.size = 0.1*np.sqrt(2)
            dynamic_obs.accel = 0.05
            dynamic_obs.max_speed = 1
        for i, dynamic_obs in enumerate(world.dynamic_obs):
            dynamic_obs.color = np.array([0., 0., 1])

        # set initial states of all agents
        agents_ctr = np.random.uniform(-4.5, +4.5, world.dim_p)
        # agents_ctr = np.array([-0.9, -0.9])
        # add Path
        path_type = 'line'
        world.start = np.array(([-2.5, -0.33/2]))
        world.path = self.set_path(path_type, world.start)
        #world.goal = world.path[1]
        world.station_num = len(world.path)
        world.station = 1

        leader_vel = np.random.uniform(0.1, 0.3)
        for i, obstacle in enumerate(world.dynamic_obs):
            obstacle.state.p_pos = world.start + np.array([i*np.random.uniform(2,4), 0])
            obstacle.state.p_ang = np.arctan2(world.path[1][1] - obstacle.state.p_pos[1], world.path[1][0] - obstacle.state.p_pos[0])
            obstacle.state.p_vel = leader_vel*np.array(world.path[1] - world.start)/np.linalg.norm(world.path[1] - world.start)
            obstacle.action.u = np.array([0.,0.])

        for agent in world.agents:
            agent.agents_ctr = agent.agents_ctr_prev = agents_ctr#world.start
            '''agent.state.p_pos = np.array([agent.agents_ctr[0]+0.1*math.cos(float(agent.name[-1])*2*np.pi/len(world.agents)),
                                          agent.agents_ctr[1]+0.1*math.sin(float(agent.name[-1])*2*np.pi/len(world.agents))])'''
            agent.state.p_pos = world.start - 2*np.array(
                [world.path[1][0] - world.start[0], world.path[1][1] - world.start[1]]) / np.linalg.norm(
                world.start - world.path)
            agent.state.p_ang = agent.ang2goal = np.arctan2(world.path[1][1] - agent.state.p_pos[1], world.path[1][0] - agent.state.p_pos[0])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_omg = 0.0
            #agent.state.p_ang = np.random.uniform(0, np.pi)
            agent.state.c = np.zeros(world.dim_c)
            agent.dis2goal = np.linalg.norm(agent.state.p_pos - world.path[1])
            agent.dis2goal_prev = None
            agent.ang2goal_prev = None
            agent.dis2obs_prev = 0
        world.goal_vect = (world.dynamic_obs[0].state.p_pos - world.agents[
                0].state.p_pos) / np.linalg.norm(world.dynamic_obs[0].state.p_pos - world.agents[0].state.p_pos)
        if num_dynamic_obs > 1:
            vect = 0.5 * np.linalg.norm(
                world.dynamic_obs[1].state.p_pos - world.dynamic_obs[0].state.p_pos)
        else:
            vect = 6 * np.linalg.norm(
                world.dynamic_obs[0].state.p_vel)
        world.goal = world.dynamic_obs[0].state.p_pos + np.multiply(world.goal_vect,vect)



    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    # collision detect
    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        if (entity1.state.p_pos[0] - entity2.state.p_pos[0]) != 0:
            ang = np.arctan((entity1.state.p_pos[1] - entity2.state.p_pos[1]) / (
                        entity1.state.p_pos[0] - entity2.state.p_pos[0]))
        else:
            ang = np.pi / 2
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = (entity1.size + entity2.size)*np.cos(abs(ang)-np.pi/4)
        if dist < dist_min:
            if 'agent' in entity1.name:
                entity1.crash += 1
            if 'agent' in entity2.name:
                entity2.crash += 1
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # return all static obstacles
    def get_static_obstacles(self, world):
        return [static_obs for static_obs in world.static_obs]

    # return all dynamic obstacles
    def get_dynamic_obstacles(self, world):
        return [dynamic_obs for dynamic_obs in world.dynamic_obs]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    '''def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False'''

    def outside_boundary(self, entity):
        # 20190711 restrict the agents in the frame
        if entity.state.p_pos[0] + entity.size > 5:
            entity.state.p_pos[0] = 5 - entity.size
            #return True
        if entity.state.p_pos[0] - entity.size < -5:
            entity.state.p_pos[0] = -5 + entity.size
            #return True
        if entity.state.p_pos[1] + entity.size > 5:
            entity.state.p_pos[1] = 5 - entity.size
            #return True
        if entity.state.p_pos[1] - entity.size < -5:
            entity.state.p_pos[1] = -5 + entity.size
            #return True
        #return False


    def agent_reward(self, agent, world):
        # 20190711
        # Agents are rewarded based on
        # 1.whether any collision happened
        # 2.the distance between formation center and desire path
        # 3.how the shape of formation is maintained
        rew = -0.3

        # parameters for collision reward
        alpha = 8
        beta = 2
        safe_dis = 15*np.linalg.norm(agent.state.p_vel)
        #print(safe_dis)
        # detect collision
        agents = self.good_agents(world)
        # penalty of collision with agents
        '''for a in agents:
            if not a.name == agent.name:
                if self.is_collision(a, agent):
                    rew -= 5'''
        static_obs = self.get_static_obstacles(world)
        dynamic_obs = self.get_dynamic_obstacles(world)
        if agent.collide:
            # compute the reward due to how close the agent to current goal
            rew -= 100*(agent.dis2goal-agent.dis2goal_prev) #* np.abs(agent.state.p_ang - agent.ang2goal)
            # compute the reward due to distance and angle to obstacles
            dis2obs = [safe_dis] * 5
            ang2obs = [0] * 1

            for o in static_obs:
                if self.is_collision(agent, o):
                    rew -= 5
                dis2obs[dis2obs.index(max(dis2obs))] = np.linalg.norm(agent.state.p_pos - o.state.p_pos) - (agent.size + o.size)
                ang2obs[ang2obs.index(max(ang2obs))] = np.arccos(np.clip(np.dot((o.state.p_pos - agent.state.p_pos), agent.state.p_vel)/(np.linalg.norm(agent.state.p_vel)*np.linalg.norm(o.state.p_pos - agent.state.p_pos)), -1, 1))

            rew -= alpha*np.exp(-beta*min(dis2obs))

            dis2leader = [safe_dis] * 5
            ang2leader = [0] * 1
            for o in dynamic_obs:
                if self.is_collision(agent, o):
                    rew -= 5
                dis2leader[dis2leader.index(max(dis2leader))] = np.linalg.norm(agent.state.p_pos - o.state.p_pos) - (agent.size + o.size)
                ang2leader[ang2leader.index(max(ang2leader))] = np.arccos(np.clip(np.dot((o.state.p_pos - agent.state.p_pos), agent.state.p_vel)/(np.linalg.norm(agent.state.p_vel)*np.linalg.norm(o.state.p_pos - agent.state.p_pos)), -1, 1))
                if 'dynamic_obs 0' in o.name:
                    if abs(min(ang2leader)) < np.pi/6:
                        rew -= alpha*np.exp(-beta * (min(dis2leader)))
                    elif abs(min(ang2leader)) < np.pi/2:
                        rew -= alpha*np.exp(-beta / (abs(0.66-min(dis2leader)*np.sin(min(ang2leader)))))
                    else:
                        rew -= alpha*np.exp(-beta / (abs(np.sin(agent.ang2goal - agent.state.p_ang))+1e-6))


        '''if agent.dis2goal < 1e-1:
            rew += 1000'''

        #print(agent.state.p_ang)
        '''if dis2goal < 1e-1:
            # if current formation center is close enough to the goal, obtain big positive reward
            rew += 1000 * (1-dis2goal) * (world.station + 1)'''

        def bound(x):
            if x < 4:
                return 0
            if x < 5:
                return np.exp(beta*(x-5.0))

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= alpha * bound(x)
        # compute the reward due to formation shape
        '''agent_idx = int(agent.name[-1])
        if agent_idx > 0:
            if agent_idx > 0 and agent_idx < len(agents)-1:
                dis2neighbor1 = np.linalg.norm(agents[agent_idx].state.p_pos - agents[agent_idx + 1].state.p_pos)
                dis2neighbor2 = np.linalg.norm(agents[agent_idx].state.p_pos - agents[agent_idx - 1].state.p_pos)
            elif agent_idx == 0:
                dis2neighbor1 = np.linalg.norm(agents[agent_idx].state.p_pos - agents[agent_idx + 1].state.p_pos)
                dis2neighbor2 = np.linalg.norm(agents[agent_idx].state.p_pos - agents[-1].state.p_pos)
            elif agent_idx == len(agents)-1:
                dis2neighbor1 = np.linalg.norm(agents[agent_idx].state.p_pos - agents[0].state.p_pos)
                dis2neighbor2 = np.linalg.norm(agents[agent_idx].state.p_pos - agents[agent_idx - 1].state.p_pos)
            #print(np.abs(dis2neighbor1 - 0.1))
            if np.abs(dis2neighbor1 - 0.1) > 1e-2:
                rew -= dis2neighbor1/2
            if np.abs(dis2neighbor2 - 0.1) > 1e-2:
                rew -= dis2neighbor2/2'''


        #print(rew)

        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5
        return rew


    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # make sure the agent is inside the window
        self.outside_boundary(agent)
        # get distance and relative angle of all entities in this agent's reference frame
        entity_dis = []
        entity_ang = []
        obs = world.static_obs + world.dynamic_obs
        for entity in obs:
            if 'dynamic_obs' in entity.name:
                entity.state.p_vel[0] = min(0.3, max(0.1, entity.state.p_vel[0]+0.01*np.random.choice([0, 1], size=1, p=[.97, .03])))
            if not entity.boundary:
                self.outside_boundary(entity)
                dis2obs = np.array([np.linalg.norm(entity.state.p_pos - agent.state.p_pos)])
                #if dis2obs < 5e-1:
                entity_dis.append(dis2obs)
                ang = np.arctan2(entity.state.p_pos[1] - agent.state.p_pos[1], entity.state.p_pos[0] - agent.state.p_pos[0])
                '''if ang < 0:
                    ang += 2 * np.pi'''
                entity_ang.append(np.array([ang - agent.state.p_ang]))
        if len(world.dynamic_obs) < 2:
            for i in range(2-len(world.dynamic_obs)):
                entity_dis.append(np.array([0]))
                entity_ang.append(np.array([0]))
        # communication of all other agents, now assume the communication graph is fully connected
        comm = []
        other_pos = []
        other_vel = []
        other_dis = []
        other_ang = []
        #print(formation_pos_x)
        comm = [world.agents[0].state.c]

        leader = world.agents[0]

        # calculate distance and angle to goal
        agent.dis2goal_prev = agent.dis2goal
        agent.ang2goal_prev = agent.ang2goal
        dis2goal = []
        ang_err = []
        vel = []
        omg = []
        agent.dis2goal = np.linalg.norm(agent.state.p_pos - world.goal)
        ang = np.arctan2(world.goal[1] - agent.state.p_pos[1], world.goal[0] - agent.state.p_pos[0])
        '''if ang < 0:
            ang += 2 * np.pi'''
        agent.ang2goal = ang
        dis2goal.append(np.array([agent.dis2goal]))
        ang_err.append(np.array([agent.ang2goal - agent.state.p_ang]))
        vel.append(np.array([np.linalg.norm(agent.state.p_vel)]))
        omg.append(np.array([np.linalg.norm(agent.state.p_omg)]))
        if agent.leader:
            #print(agent.name)
            #print(agent.state.p_vel)
            return np.concatenate(dis2goal + ang_err + vel + omg + other_dis + other_ang + entity_dis + entity_ang)
        else:
            #print(np.concatenate(dis2goal + ang_err + vel + omg + entity_dis + entity_ang))
            print(entity_dis[4])
            return np.concatenate(dis2goal + ang_err + vel + omg + entity_dis + entity_ang)

        # todo
        # the return observation should include
        # 1.the distance and relative angle between the agent and its neighbor (displacement-based)
        # 2.the distance from the goal
        # 3.the possible moving area observed by this agent, which avoid itself from collision
        # (communicate with neighbor agents to update this area)

    def done(self, agent, world):
        dis = np.linalg.norm(agent.state.p_pos - world.goal)
        if len(world.dynamic_obs) > 1:
            vect_offset = 0.5 * np.linalg.norm(
                world.dynamic_obs[1].state.p_pos - world.dynamic_obs[
                    0].state.p_pos)
        else:
            vect_offset = 6 * np.linalg.norm(
                world.dynamic_obs[0].state.p_vel)
        if dis < 4e-1:
            #return True
            if world.station == world.station_num:
                return True
            else:
                world.station += 1
                world.goal += np.array([vect_offset, 1]) * world.goal_vect
        else:
            if world.station < world.station_num:
                world.goal = world.dynamic_obs[0].state.p_pos + np.array([vect_offset, 1]) * world.goal_vect
        return False


