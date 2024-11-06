# time: 2023/10/30 14:57
# author: YanJP
# import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import para
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
# from ppo_discrete import PPO_discrete,HPPO
# from ppo_continuous import PPO_continuous
from ppo_discrete_multi_actions import PPO_discrete
import envs
from tqdm import tqdm
from Draw_pic import *
import time
# from baseline import *
seed = para.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
import pickle
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
def save_state_norm(state_norm):
    filepath="runs/model/state_norm/"+curr_time+'state_norm.pkl'
    with open(filepath, 'wb') as file:
        pickle.dump(state_norm, file)

def load_state_norm():
    file=get_file_model(folder_path="runs/model/state_norm")
    file = "runs/model/state_norm/" + file
    with open(file, 'rb') as file_name:
        s_n = pickle.load(file_name)
    return s_n



def write_power(powers,evaluate_o,powersum,T_delay,snr):
    with open('runs/res.txt', 'a+') as F:
        F.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        F.write("PowerSum:" + str(powersum) + "       power_list:" + str(powers) + "\n")
        F.write("Compress ratio:"+str(evaluate_o)+"\n")
        F.write("T_delay:" + str(T_delay) + "\n")
        F.write("SNR_dB:" + str(snr) + "\n\n")

def get_file_model(folder_path = "runs/model/"):
    # folder_path = "runs/model/"  # 需要修改成你想要操作的文件夹路径
    folder_u=folder_path
    file_list = [f for f in os.listdir(folder_u) if os.path.isfile(os.path.join(folder_u, f))]
    sorted_files = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(folder_u, x)), reverse=True)
    up_file = sorted_files[0]

    return up_file
def main(args, time, seed):
    env = envs.env_()
    # env_evaluate = envs.env_()
    args.state_dim = env.observation_space
    args.action_dim = para.action_dim
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    # print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))
    replay_buffer = ReplayBuffer(args)
    rewards=[]
    PA_post_rewards=[]
    PA_post_ma_rewards = []
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    agent=PPO_discrete(args)
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    for total_steps in tqdm(range(1,args.max_train_steps+1)):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        episode_steps = 0
        done = False
        episode_rewards = []
        # PA_post_reward=[]
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, info = env.step(a)
            episode_rewards.append(r)
            if args.use_state_norm:
                s_ = state_norm(s_)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            if sum(info)!=0:
                replay_buffer.update_r(info)
            s = s_
            total_steps += 1
            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
            #     # Evaluate the policy every 'evaluate_freq' steps
            #     if total_steps % args.evaluate_freq == 0:
            #         evaluate_num += 1
            #         evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
            #         evaluate_rewards.append(evaluate_reward)
            #         print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
        ep_reward=sum(episode_rewards)
        PA_post_reward=sum(info)
        print("ep_PA_post_reward:",PA_post_reward)
        # print("ep_reward:",ep_reward)

        rewards.append(ep_reward)
        PA_post_rewards.append(PA_post_reward)
        # if ma_rewards:
        #     ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        # else:
        #     ma_rewards.append(ep_reward)
    agent.savemodel(time)
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'PA_post_rewards': PA_post_rewards}
def evaluate_policy( env, agent):
    times = 1
    evaluate_reward = 0
    for _ in range(times):
        s  = env.reset()
        episode_steps = 0
        done = False
        episode_rewards = []
        PA_post_episode_rewards=[]
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, info = env.step(a)
            s = s_
            episode_rewards.append(r)
            if sum(info)!=0:
                PA_post_episode_rewards=info
        print("ACC:{}",episode_rewards)
        print("PA_post_ACC:",PA_post_episode_rewards)
        ACC,PA_post_ACC=sum(episode_rewards)/(para.K),sum(PA_post_episode_rewards)/(para.K)
        # print("ACC:",ACC)
        # print("PA_post_ACC:",PA_post_ACC)

    return ACC,PA_post_ACC,env.final_ps
def test(args):
    env = envs.env_()
    args.state_dim = env.observation_space
    args.action_dim = para.action_dim
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    evaluate_num = 0  # Record the number of evaluations

    agent = PPO_discrete(args)
    model_ = get_file_model()
    path1 = 'runs/model/' + model_
    agent.load_model(path1)
    # start = time.perf_counter()
    ACCs=0
    PA_post_ACCs=0
    for total_steps in range(1,args.max_test_steps+1):
        Accs,PA_post_ACC, evaluate_power = evaluate_policy(  env, agent)
        print("num:{} \t ACC:{} \t PA_post_ACC:{} ".format(total_steps, Accs, PA_post_ACC))
        ACCs+=Accs
        PA_post_ACCs+=PA_post_ACC
    # end = time.perf_counter()  # 记录结束时间
    # duration = end - start  # 计算运行时间
    # print("程序运行时间为：", duration)
    final_test_acc=ACCs/args.max_test_steps
    final_PA_post_test_acc=PA_post_ACCs/args.max_test_steps
    print("final_test_acc:",final_test_acc)
    print("final_PA_post_test_acc:",final_PA_post_test_acc)

    return final_test_acc,final_PA_post_test_acc
        # ep_reward=sum(episode_rewards)
        # print("ep_reward:",ep_reward)
        # rewards.append(ep_reward)

def test_ppo():
    Nc=np.array([150,160,170,180,190,200])-60
    for i,nc in enumerate(Nc):
        para.N_c=nc
        ppo = test(args)
        print(ppo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(5.0e3), help=" Maximum number of training steps")
    parser.add_argument("--max_test_steps", type=int, default=int(3), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=10, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size") #2048
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")  # 3e-5
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.04, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    train=True
    # train=False
    train_log_dir='runs/rewards/'+curr_time
    if train:
        res_dic=main(args, curr_time, seed=seed)
        np.save(train_log_dir + '_reward.npy', np.array([res_dic['rewards'],res_dic['PA_post_rewards']])/para.K)
        plot_rewards(np.array(res_dic['rewards'])/para.K,np.array(res_dic['PA_post_rewards'])/para.K,curr_time,path='runs/pic')
    else:
        test(args)

