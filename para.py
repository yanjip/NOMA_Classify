# time: 2024/1/26 21:20
# author: YanJP
import numpy as np
KHz=1e3
MHz=1e6

N=4 # 基站数量
K=10 # 用户数量
p_max=0.2

Ws=np.random.choice(np.array([1,2,3,4,5])*MHz,size=N)
num_videos=15  # 一共有15份视频
cachelen=6  # 每个基站的视频存储长度
state_dim =5    # 自己基站的视频缓存（设定所有基站共享视频，但有时延）、每个视频的请求热度（Zipf给出，用户按照这个概率请求视频）、
action_dim=3   # 功率、调度决策、SCR
d0=5e6
t_max=0.08
sigma2=6e-12
ACC=0.8  # 后面再改
def Succe_Prob(power,ratio,lar_fad,W,interfence):
    phy=d0/(W*t_max)
    # xi=power/(N0_dBm*W)
    xi=power*lar_fad/(sigma2+interfence)
    delta=1
    if power!=0:
        x=(2**(phy*ratio)-1)/(xi*delta)
        prob=np.exp(-x**2/2)
    else: prob=0
    return prob

def get_hot():
    h=np.random.uniform(0.1,0.9,size=num_videos)
    nor_hot=h/sum(h)
    return nor_hot

a=1.8
def get_hot_zipf():
    # 生成Zipf分布的概率
    # a =1.8  # Zipf分布的参数，可以调整   2.5
    zipf_probs = np.random.zipf(a, num_videos)
    # 归一化概率，使其总和为1
    zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)
    # 打印生成的概率
    # print("生成的Zipf分布概率：", zipf_probs_normalized)
    return zipf_probs_normalized

def ini_video():
    videos_c=[]
    for i in range(3):
        video_cache = np.random.choice(np.arange(num_videos), size=cachelen, replace=False)
        videos_c.append(video_cache)
    all_video_hot = get_hot_zipf()

    return videos_c,all_video_hot

videos_c,all_video_hot=ini_video()

train=True
test_times=40



def get_large_fading():
    distances = np.random.uniform(2, 5, (N,K))*100   # Distances in meters
    return distances**(-2)
def get_small_fading():
    # 生成 Rayleigh 衰落参数 γ
    h= np.random.rayleigh(scale=1,size=(N,K))
    # large_fading=get_large_fading()
    return h
large_fading=get_large_fading()
h=get_small_fading()

if __name__ == '__main__':

    (get_hot_zipf())

