# time: 2024/11/4 14:59
# author: YanJP
import numpy as np

import para
from train import test
from Baselines import *
import argparse
from Draw_pic import *
def proposed():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(4.0e3), help=" Maximum number of training steps")
    parser.add_argument("--max_test_steps", type=int, default=int(1), help=" Maximum number of training steps")
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
    ans_b1,ans_proposed=test(args)
    return ans_b1,ans_proposed
def compare_multi():
    x=5 # 8次测试
    ans=np.zeros((5,x))
    for i in range(x):
        # para.seed+=i*2
        # para.p_max = np.maximum(0.1, np.random.normal(0.1, 2, para.K))
        # para.Ws = np.random.choice(np.array([1, 2, 3, 4, 5]) + 4, size=para.N) * para.MHz
        # para.large_fading = para.get_large_fading()
        # h = para.get_small_fading()
        # para.H = np.sort(para.large_fading * h, axis=1)
        # para.h = para.H / para.large_fading
        ans_b1, ans_proposed = proposed()
        b2 = Baseline2()
        ans_b2 = b2.get_acc()
        print("Baseline2:", ans_b2)

        b3 = Baseline3()
        ans_b3 = b3.get_acc()
        print("Baseline3:", ans_b3)

        b4 = Baseline4()
        ans_b4 = b4.get_acc()
        print("Baseline4:", ans_b4)
        ans[:,i]=ans_proposed,ans_b1,ans_b2,ans_b3,ans_b4
    return ans.mean(axis=1)
def change_tmax():
    # ts=np.array([0.05,0.10,0.15,0.20,0.25])
    ts=np.array([0.05,0.08,0.11,0.14,0.17])+0.02
    ans=np.zeros((5,len(ts)))
    for i,t in enumerate(ts):
        para.t_max=t
        res=compare_multi()
        ans[:,i]=res
    plot_tmax(ts,ans)
    # np.save("runs/datas/res_tmax_11_7.npy",ans)
    pass
def change_Pth():
    # ts=np.array([0.05,0.10,0.15,0.20,0.25])
    Pths=np.array([0.95,0.96,0.97,0.98,0.99])
    ans=np.zeros((5,len(Pths)))
    for i,p in enumerate(Pths):
        para.Prob_th=p
        res=compare_multi()
        ans[:,i]=res
    plot_Pth(Pths,ans)
    np.save("runs/datas/Pth_11_7.npy",ans)
if __name__ == '__main__':
    # ans=compare_multi()
    # print(ans)
    # change_tmax()

    change_Pth()

    pass