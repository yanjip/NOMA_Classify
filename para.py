# time: 2024/1/26 21:20
# author: YanJP
import numpy as np
KHz=1e3
MHz=1e6
seed=10
np.random.seed(seed)
N=4 # 基站数量
K=16 # 用户数量
p_max=np.maximum(0.1,np.random.normal(0.1,2,K))
max_action=1
Ws=np.random.choice(np.array([1,2,3,4,5])+4,size=N)*MHz

state_dim =2*N    # 所有基站对用户k的信道增益、 interference
action_dim=2   # 功率、调度决策、SCR
output_dims=[N,10]
d0=6e6
t_max=0.15
Prob_th=0.98
sigma2=6e-12
N0 = 3.981e-21  #-174dBm==3.981e-21
# N0 = 3.981e-18  #-174dBm==3.981e-21

W_n=1
SCRs=[]  # 0-9的索引整数
scr_range=np.arange(0.1,1.1,0.1)
# ACC=0.8  # 后面再改
fit_params=np.load('fit_params.npy')
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



a=1.8


# def ini_video():
#     videos_c=[]
#     for i in range(3):
#         video_cache = np.random.choice(np.arange(num_videos), size=cachelen, replace=False)
#         videos_c.append(video_cache)
#     all_video_hot = get_hot_zipf()
#
#     return videos_c,all_video_hot
#
# videos_c,all_video_hot=ini_video()

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
H=np.sort(large_fading*h, axis=1)
h=H/large_fading



def dBm2wat(power_dbm):
    power_watt = np.power(10, (power_dbm - 30) / 10)
    return power_watt
def wat2dBm(power_watt):
    power_dbm = 30 + 10 * np.log10(power_watt)
    return power_dbm
if __name__ == '__main__':
    print(H)
    # (get_hot_zipf())
    print(len(fit_params))
    print(dBm2wat(-174))
    pass
