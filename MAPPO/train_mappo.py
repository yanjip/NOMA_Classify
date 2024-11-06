# time: 2024/9/29 21:03
# author: YanJP
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

import para
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_model import MAPPO_agent
import envs
from tqdm import tqdm
import datetime
from Draw_pic import *
import os
import pickle
seed=para.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def get_file_model(folder_path = "runs/model/"):
    # folder_path = "runs/model/"  # 需要修改成你想要操作的文件夹路径
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    sorted_files = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    latest_file = sorted_files[0]
    return latest_file

def save_state_norm(state_norm):
    filepath="runs/model/state_norm/"+curr_time+'state_norm.pkl'
    with open(filepath, 'wb') as file:
        pickle.dump(state_norm, file)
def load_state_norm():
    file=get_file_model(folder_path="../runs/model/state_norm")
    file = "runs/model/state_norm/" + file
    with open(file, 'rb') as file_name:
        s_n = pickle.load(file_name)
    return s_n

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        # Create env
        self.env = envs.env_() # Discrete action space
        self.args.N = self.env.n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        self.state_norm = Normalization(shape=para.state_dim)  # Trick 2:state normalization

        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_agent(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def norm_state(self, obs):
        res = []
        for o in obs:
            nor_o = self.state_norm(o)
            res.append(nor_o)
        return res
    def run(self, time):
        rewards = []
        ma_rewards = []  # 记录所有回合的滑动平均奖励
        evaluate_num = -1  # Record the number of evaluations
        for total_steps in tqdm(range(1, args.max_train_steps + 1)):
            # while self.total_steps < self.args.max_train_steps:
            if total_steps % self.args.evaluate_freq == 0:
                # self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
            #-------------每个eposide-----------------
            ep_reward, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode

            if total_steps%20==0:
                print("ep_reward:", ep_reward)
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, total_steps)  # Training
                self.replay_buffer.reset_buffer()

        # self.evaluate_policy()
        path = 'runs/model/ppo_' + time + '.pth'
        torch.save(self.agent_n.actor.state_dict(), path)
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)
        save_state_norm(self.state_norm)
        return {'episodes': range(len(rewards)), 'rewards': rewards, 'ma_rewards': ma_rewards}


    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        # self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        # np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n = self.env.reset() #[[...], [....],   ] 都是列表
        # obs_n=self.norm_state(obs_n)
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        done_n=False
        episode_step = -1
        episode_rewards=[]
        for k in range(para.K):
            episode_step += 1
            parameter_action, raw_act, parameter_logp_t = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _ = self.env.step(parameter_action)
            episode_reward += sum(r_n)   #reward也是相同的三个值
            # obs_next_n = self.norm_state(obs_next_n)

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, parameter_action, parameter_logp_t, r_n, done_n)

            obs_n = obs_next_n
            if done_n:
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1
    def test(self,):
        model = get_file_model()
        path = 'runs/model/' + model
        self.agent_n.load_model(path)
        self.state_norm=load_state_norm()
        all_r=[]
        for i in range(1, 30):
            #-------------每个eposide-----------------
            ep_reward, episode_steps = self.run_episode_mpe(evaluate=True)  # Run an episode
            print("ep_reward:", ep_reward)
            all_r.append(ep_reward)
        print("avg_ep_reward:", sum(all_r)/len(all_r))
        return sum(all_r)/len(all_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(2.0e3), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=para.K, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=para.seed)
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    train=True
    # train=False
    if train:
        res_dic=runner.run(curr_time)

        train_log_dir='runs/rewards/'+curr_time
        np.save(train_log_dir + '_reward.npy', np.array(res_dic['rewards']))
        plot_rewards(res_dic['rewards'], curr_time, path='../runs/pic')
    else:
        runner.test()