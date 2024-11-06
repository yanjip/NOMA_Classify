# time: 2023/12/23 21:36
# author: YanJP
import numpy as np
import matplotlib.pyplot as plt
import random
import para
import datetime
from matplotlib.font_manager import FontProperties  # 导入字体模块

# 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
def chinese_font():
    try:
        font = FontProperties(
            # 系统字体路径
            fname='C:\\Windows\\Fonts\\方正粗黑宋简体.ttf', size=14)
    except:
        font = None
    return font
def process_res(res):
    proposed=res[0,:]
    b1=res[ 1,:]
    b2=res[ 2,:]
    b3=res[ 3,:]
    b4=res[ 4,:]
    return proposed,b1,b2,b3,b4
def plot_rewards(rewards,rewards2, time=None,path=None,save=True):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=15)
    plt.xlabel('Episodes', fontsize=17, fontweight='bold', labelpad=-1)
    plt.ylabel('Reward', fontsize=17, fontweight='bold', labelpad=-1)
    plt.grid(linestyle="--", color="gray", linewidth="0.3", axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    s_r1 = smooth(rewards)
    plt.plot(rewards, alpha=0.3, color='c',linewidth='2.5')
    plt.plot(s_r1, linewidth='2.0',label='Baseline 1',)

    s_r2 = smooth(rewards2)
    plt.plot(rewards2, alpha=0.3, color='orange',linewidth='2.5')
    plt.plot(s_r2, linewidth='2.0', label='Proposed', color='orange')
    plt.legend() #prop={'weight': 'light'}
    if time is None:
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if save:
        plt.savefig(f'{path}_{time}.png')
    # plt.savefig("runs/baseline/SAC_convergence_" + a, dpi=600, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig("runs/baseline/SAC_convergence_" + a+'.eps',format='eps', dpi=600, bbox_inches='tight', pad_inches=0.01)

    plt.show()
def plot_rewards_file(rewards,rewards2, time=None,path=None,save=True):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=15)
    plt.xlabel('Episodes', fontsize=17, fontweight='bold', labelpad=-1)
    plt.ylabel('Reward', fontsize=17, fontweight='bold', labelpad=-1)
    plt.grid(linestyle="--", color="gray", linewidth="0.3", axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


    s_r2 = smooth(rewards2)
    # plt.plot(rewards2, alpha=0.3, color='orange',linewidth='2.5')
    plt.plot(s_r2, linewidth='2.0', label='Proposed Algorithm 2', color='orange')

    s_r1 = smooth(rewards)
    # plt.plot(rewards, alpha=0.3, color='c',linewidth='2.5')
    plt.plot(s_r1, linewidth='2.0',label='Baseline 1',)
    plt.legend() #prop={'weight': 'light'}
    plt.ylim(0.3)
    if time is None:
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if save:
        plt.savefig(f'{path}_{time}.png')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # plt.savefig("runs/baseline/SAC_convergence_" + a, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.savefig("runs/baselines/convergence_" + a+'.eps',format='eps', dpi=600, bbox_inches='tight', pad_inches=0.01)

    plt.show()
def plot_tmax(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'
    proposed, b1, b2, b3,b4 = process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed Scheme',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b2, marker='*', markersize=8, label='Baseline 2',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b4, marker='s', markersize=8, label='Baseline 4',markerfacecolor='none')  # 使用三角形节点
    plt.rc('font', size=17)
    plt.legend(loc='lower right', ncol=1)
    # plt.ylim(2)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('t (s)', fontsize=17,fontweight='bold',labelpad=-2)
    plt.ylabel('Classification Accuracy', fontsize=16,fontweight='bold',labelpad=-10)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # plt.savefig("runs/baselines/tmax" + a, dpi=600, bbox_inches='tight', pad_inches=0.01)
    # 显示图形
    plt.show()


def plot_bar(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', size=13)
    plt.grid(linestyle="--", color="gray", linewidth="0.5", axis="both")
    proposed, b1, b2, b3 = process_res(res)
    categories =x
    x = np.arange(len(categories))
    # 使用plt.bar()替代plt.plot()
    width = 0.2  # 设置柱状图的宽度

    plt.bar([i + 0 * width for i in x], b3, width=width, label='Baseline 1', color='#2ca02c',alpha=0.8)

    plt.bar([i + 1 *width for i in x], b1, width=width, label='Baseline 2',color='#1f77b4',alpha=0.9)
    plt.bar([i + 2 *width for i in x], proposed, width=width, label='Proposed Scheme',color='#ff7f0e',alpha=1)

    # plt.bar([i + 2 * width for i in x], b2, width=width, label='Baseline 2', alpha=0.7)
    plt.ylim(0.01, 0.16)


    plt.xticks([i + 1.5 * width for i in x], categories)  # 调整x轴刻度位置
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(r'The computing resource of BS (Gbit/second)', fontsize=17, fontweight='bold', labelpad=0)
    plt.ylabel('Latency (s)', fontsize=17, fontweight='bold', labelpad=1)
    plt.legend(ncol=1)

    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # plt.savefig("runs/baseline/computing" + a, dpi=700, bbox_inches='tight', pad_inches=0.01)
    plt.show()

# 用于平滑曲线，类似于Tensorboard中的smooth
def smooth(data, weight=0.97):
    '''
    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

if __name__ == '__main__':
    rewards=np.load('runs/rewards/2024_11_01-21_21_40_reward.npy')
    plot_rewards_file(rewards[0],rewards[1],path='runs/baselines/',save=False)
    pass