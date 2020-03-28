import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.ddpg import DDPGAgentTrainer
from maddpg.trainer.maRDPG import MARDPGAgentTrainer
from maddpg.trainer.dqn import DQNAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="overtaking", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=250, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="dqn", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--network", type=str, default="MLP", help="define neural network type")
    parser.add_argument("--trajectory_size", type=int, default=25)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='Test', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="C:/Users/xinyouqiu/Desktop/北京清華/科研/開題/仿真環境/maddpg/policy/model_dqn.ckpt", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="C:/Users/xinyouqiu/Desktop/北京清華/科研/開題/仿真環境/maddpg/trainResult/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="C:/Users/xinyouqiu/Desktop/北京清華/科研/開題/仿真環境/maddpg/trainResult/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units*2, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def lstm_model(obs, trajectory_size, num_inputs, num_outputs, scope, reuse=False, num_units=300, rnn_cell=None):

    obs = tf.reshape(obs, [-1, trajectory_size, num_inputs])
    sequence = tf.placeholder(tf.float32, [None, trajectory_size, num_inputs])
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def linear(input):
        out = tf.reshape(input, [-1, num_inputs])
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        #out = layers.fully_connected(out, num_outputs=300, activation_fn=tf.nn.relu)
        #out = tf.reshape(out, [-1, trajectory_size, 128])
        return out
    outputs = []

    with tf.variable_scope(scope, reuse=reuse):
        softmax_w = tf.get_variable(name="weight", shape=[19, num_outputs])
        softmax_b = tf.get_variable(name="bias", shape=[num_outputs])
        for step in range(trajectory_size):
            outputs.append(obs[:, step, :])
        outputs = tf.split(tf.concat(outputs,1), 19, 1)
        output, state = tf.contrib.rnn.static_rnn(tf.contrib.rnn.BasicLSTMCell(19), outputs, dtype=tf.float32)
        return tf.matmul(output[-1], softmax_w) + softmax_b

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.constraint, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.constraint, done_callback=scenario.done)
    return env

def get_trainers(env, num_adversaries, his_shape_n, obs_shape_n, arglist):
    trainers = []
    if arglist.network=="MLP":
        model = mlp_model
        if arglist.good_policy=="ddpg":
            trainer = DDPGAgentTrainer
            trainers.append(trainer(model, obs_shape_n, [env.action_space[0]], env.n, arglist))
        elif arglist.good_policy=="dqn":
            trainer = DQNAgentTrainer
            trainers.append(trainer(model, obs_shape_n, [env.action_space[0]], env.n, arglist,
                                    local_q_func=(arglist.adv_policy == 'ddpg')))
        else:
            trainer = MADDPGAgentTrainer
            for i in range(num_adversaries):
                trainers.append(trainer(
                    "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                    local_q_func=(arglist.adv_policy=='ddpg')))
            for i in range(num_adversaries, env.n):
                trainers.append(trainer(
                    "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                    local_q_func=(arglist.good_policy=='ddpg')))

    else:
        model = lstm_model
        trainer = MARDPGAgentTrainer
        for i in range(num_adversaries):
            trainers.append(trainer(
                "agent_%d" % i, model, his_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy == 'ddpg')))
        for i in range(num_adversaries, env.n):
            trainers.append(trainer(
                "agent_%d" % i, model, his_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        if arglist.good_policy == "ddpg":
            # only one agent's observation is considered in ddpg
            obs_shape_n = [env.observation_space[0].shape]
            his_shape_n = [((env.observation_space[0].shape[0] + 3) * arglist.trajectory_size,)]
        else:
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            his_shape_n = [((env.observation_space[i].shape[0]+3)*arglist.trajectory_size,) for i in range(env.n)]
        con_shape_n = [env.constraint_space[0].shape] #2020/02/20
        num_adversaries = min(env.n, arglist.num_adversaries)
        episode_step = [0]
        final_ep_steps = []
        episode_done = [0]
        final_ep_done = []
        train_step = 0
        trainers = get_trainers(env, num_adversaries, his_shape_n, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        episode_crash = [0]  # sum of crashes for all agents
        final_ep_crash = []  # sum of crashes for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()

        obs_n = env.reset()
        '''def his_padding(his):
            his = np.concatenate((his, np.zeros(475-len(his))))
            return his
        action_pre = [np.array(np.random.uniform(0,1,3))] * env.n # an initial action value for episode step = 0
        his_pre = [] * env.n
        for i in range(env.n):
            his_pre.append(np.concatenate((action_pre[i], obs_n[i])))
        his_n_a = [] * env.n
        his_n_c = [] * env.n
        for i in range(env.n):
            his_n_a.append(his_pre[i])
            his_n_c.append(his_pre[i])'''
        t_start = time.time()
        final_reward_prev = None
        print('Starting iterations...')


        while True:
            if arglist.network == "MLP":
                # get action
                if arglist.good_policy == "maddpg":
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                else:
                    action_n = [trainers[0].action(obs_n[obs],train_step) for obs in range(len(obs_n))]
                #print(action_n)
                # environment step
                new_obs_n, rew_n, done_n, info_n, crash_n = env.step(action_n)
                episode_step[-1] += 1
                done = all(done_n)
                terminal = (episode_step[-1] >= arglist.max_episode_len)
                # collect experience
                if arglist.good_policy == "ddpg":
                    for i in range(len(obs_n)):
                        trainers[0].experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                else:
                    for i, agent in enumerate(trainers):
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                for i, crash in enumerate(crash_n):
                    episode_crash[-1] += crash

                for i, done in enumerate(done_n):
                    episode_done[-1] += done

                if done or terminal:
                    obs_n = env.reset()
                    episode_step.append(0)
                    episode_rewards.append(0)
                    episode_crash.append(0)
                    episode_done.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])
            else:
                # get action
                action_n = [agent.action(his_padding(his)) for agent, his in zip(trainers, his_n_a)]
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience
                new_his_n = [] * env.n
                for i, agent in enumerate(trainers):
                    # his_t = [his_t-1, obs_t, action_t]
                    '''if episode_step > arglist.trajectory_size-1:
                        new_his_n[i][0:len(his_n_a[i])-len(his_n_a[i])//arglist.trajectory_size] = his_n_a[i][len(his_n_a[i])//arglist.trajectory_size:]
                        new_his_n[i][len(his_n_a[i])-len(his_n_a[i])//arglist.trajectory_size:] = np.concatenate((new_obs_n[i], action_n[i]))[:]
                    else:
                        new_his_n[i][len(his_n_a[i]) // arglist.trajectory_size*episode_step:len(his_n_a[i]) // arglist.trajectory_size*(episode_step+1)] = np.concatenate(
                            (new_obs_n[i], action_n[i]))[:]'''
                    new_his_n.append(np.concatenate((his_n_a[i], np.concatenate((action_n[i], new_obs_n[i])))))
                    if len(new_his_n[i]) > arglist.trajectory_size*19:
                        new_his_n[i] = new_his_n[i][19:]
                    # store transition [h_t-1. a_t, r_t, h_t] into replay buffer
                    agent.experience(his_padding(his_n_c[i]), action_n[i], rew_n[i], his_padding(new_his_n[i]), done_n[i], terminal)
                his_n_a = new_his_n
                his_n_c = new_his_n
                '''his_n_c[:][0:-1 - 2] = his_n_a[:][3:]
                his_n_c[:][-1 - 2:] = action_n[:][:]'''
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                if done or terminal:
                    obs_n = env.reset()
                    action_pre = [np.array(
                        np.random.uniform(0, 1, 3))] * env.n  # an initial action value for episode step = 0
                    his_pre = [] * env.n
                    for i in range(env.n):
                        his_pre.append(np.concatenate((action_pre[i], obs_n[i])))
                    his_n_a = [] * env.n
                    his_n_c = [] * env.n
                    for i in range(env.n):
                        his_n_a.append(his_padding(his_pre[i]))
                        his_n_c.append(his_padding(his_pre[i]))
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                '''if train_step == 10:
                    break'''
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                final_reward = np.mean(episode_rewards[-arglist.save_rate:])
                final_step = np.mean(episode_step[-arglist.save_rate:])
                final_crash = np.mean(episode_crash[-arglist.save_rate:])
                final_done = np.mean(episode_done[-arglist.save_rate:])
                if not final_reward_prev:
                    final_reward_prev = final_reward
                else:
                    if final_reward > final_reward_prev:
                        U.save_state(arglist.save_dir, saver=saver)
                        final_reward_prev = final_reward
                        print("model saved time: {}".format(round(time.time()-t_start, 3)))
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, mean episode step: {},  mean episode crash: {}, time: {}".format(
                        train_step, len(episode_rewards), final_reward, final_step, final_crash, round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), final_reward,
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(final_reward)
                final_ep_steps.append(final_step)
                final_ep_crash.append(final_crash)
                final_ep_done.append(final_done)
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_dqn.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards_dqn.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                step_file_name = arglist.plots_dir + arglist.exp_name + '_steps_dqn.pkl'
                with open(step_file_name, 'wb') as fp:
                    pickle.dump(final_ep_steps, fp)
                crash_file_name = arglist.plots_dir + arglist.exp_name + '_crashes_dqn.pkl'
                with open(crash_file_name, 'wb') as fp:
                    pickle.dump(final_ep_crash, fp)
                done_file_name = arglist.plots_dir + arglist.exp_name + '_done_dqn.pkl'
                with open(done_file_name, 'wb') as fp:
                    pickle.dump(final_ep_done, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    #train(arglist)
    '''graph = tf.get_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph("/home/shinyochiu/maddpg/policy/model.ckpt.meta")
    saver.restore(sess, "/home/shinyochiu/maddpg/policy/model.ckpt")
    tf.train.write_graph(sess.graph_def, '.', '/home/shinyochiu/maddpg/policy/graph.pb', as_text=False)
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/shinyochiu/maddpg/policy/graph.pb")
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)'''
    import pprint, pickle
    import matplotlib.pyplot as plt
    rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_2.pkl'
    pkl_file1 = open(rew_file_name, 'rb')
    rewdqn_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_dqn.pkl'
    pkl_file2 = open(rewdqn_file_name, 'rb')
    crash_file_name = arglist.plots_dir + arglist.exp_name + '_crashes_2.pkl'
    pkl_file3 = open(crash_file_name, 'rb')
    crashdqn_file_name = arglist.plots_dir + arglist.exp_name + '_crashes_dqn.pkl'
    pkl_file4 = open(crashdqn_file_name, 'rb')
    step_file_name = arglist.plots_dir + arglist.exp_name + '_steps_formation.pkl'
    pkl_file5 = open(step_file_name, 'rb')
    stepdqn_file_name = arglist.plots_dir + arglist.exp_name + '_steps_dqn.pkl'
    pkl_file6 = open(stepdqn_file_name, 'rb')
    rew = pickle.load(pkl_file1)
    rew_dqn = pickle.load(pkl_file2)
    crash = pickle.load(pkl_file3)
    crash_dqn = pickle.load(pkl_file4)
    steps = pickle.load(pkl_file5)
    steps_dqn = pickle.load(pkl_file6)
    pprint.pprint(min(crash))
    pprint.pprint(min(crash_dqn))
    pprint.pprint(steps[crash.index(min(crash))])
    pprint.pprint(steps_dqn[crash_dqn.index(min(crash_dqn))])

    x=list(range(1,len(steps)+1))
    plt.plot(rew[0:len(steps)], label='DDPG')
    plt.plot(rew_dqn[0:len(steps)], label='DQN')
    #plt.bar(x, crash_dqn[0:len(steps)], label='DQN', align="center", color='lightsteelblue')
    #plt.bar(x, crash[0:len(steps)], label='DDPG', align="edge")
    plt.xlabel('number of training episodes (x1000)')
    plt.ylabel('mean episode rewards')
    plt.legend()
    plt.show()
    pkl_file1.close()
    pkl_file2.close()
    pkl_file3.close()
    pkl_file4.close()
