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
    proposed=res[:,0]
    b1=res[:,1]
    b2=res[:,2]
    b3=res[:,3]-0.029
    return proposed,b1,b2,b3
def plot_rewards(rewards,time,  path=None,):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=15)
    plt.xlabel('Epsiodes', fontsize=17, fontweight='bold', labelpad=-1)
    plt.ylabel('Reward', fontsize=17, fontweight='bold', labelpad=-1)
    plt.grid(linestyle="--", color="gray", linewidth="0.3", axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    s_r1 = smooth(rewards)
    plt.plot(rewards, alpha=0.5, color='c')
    plt.plot(s_r1, linewidth='1.5', )
    # plt.plot(s_r2,linewidth='1.5', label='clipped probability ratio=0.5')
    # plt.ylim(-1)
    # plt.legend()
    a = time
    plt.savefig(f"{path}/{a}_power.png")
    plt.show()
def plot_BW(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    proposed, b1, b2, b3,b4 = process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed Scheme')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 2',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 4',markerfacecolor='none')  # 使用三角形节点
    plt.rc('font', size=17)
    plt.legend(loc='lower right', ncol=1)
    # plt.ylim(2)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('Bandwidth (MHz)', fontsize=17,fontweight='bold',labelpad=-2)
    plt.ylabel('QoE', fontsize=17,fontweight='bold',labelpad=-10)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/new_baseline/Bandwidth" + a, dpi=600, bbox_inches='tight', pad_inches=0.01)
    # 显示图形
    plt.show()




# 用于平滑曲线，类似于Tensorboard中的smooth
def smooth(data, weight=0.9):
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
    pass