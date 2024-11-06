# time: 2024/1/26 21:16
# author: YanJP
import numpy as np
import para
import fitting
import PA_func
class Basestation():
    def __init__(self,id):
        self.id=id
        self.W=para.Ws[id]
        self.interference=0
        self.H=para.H[id]
        self.large_h=para.large_fading[id]
        self.small_h=para.h[id]
        self.UDs=[]
    def get_power(self,):
        int_arr = self.UDs[:,0].astype(int)
        Hs= self.H[int_arr]
        para.SCRs=self.UDs[:,1].astype(int)
        pmins=self.UDs[:,2]
        ps,acc=PA_func.PA_algor(Hs,pmins,para.p_max[int_arr])
        return ps,acc

    def get_ini_obs(self,k):
        H=self.large_h[k]*self.small_h[k]
        self.obs=np.array([self.W/1e6,self.interference,self.large_h[k],self.small_h[k],H])
        # self.obs=np.concatenate((self.interference,self.large_h[k],self.small_h[k],H),axis=0)
        return self.obs

def reset_env():
    para.large_fading = para.get_large_fading()
    h = para.get_small_fading()
    para.H = np.sort(para.large_fading * h, axis=1)
    para.h = para.H / para.large_fading
    para.p_max = np.maximum(0.1, np.random.normal(0.1, 2, para.K))
    para.Ws = np.random.choice(np.array([1, 2, 3, 4, 5]) + 4, size=para.N) * para.MHz

class env_():
    def __init__(self):
        self.n=para.N
        self.action_space=para.action_dim
        self.observation_space=para.state_dim  #每个智能体的空间维度
        # self.UserAll=trans.generateU()
        self.reward=0
        self._max_episode_steps=para.K
        self.BSs= [Basestation(i) for i in range(para.N)]
        self.ACCs=np.zeros(para.K)
        # self.h=para.h
        # self.min_simis=para.min_sims
        # self.salency=para.salency
        # self.request_tiles=np.random.randint(para.N_fov_low,para.N_fov)
        pass
    def get_all_obs(self,k):
        obs=[]
        for n in range(para.N):
            obs.append(self.BSs[n].H[k]*1e6)
        for n in range(para.N):
            obs.append(self.BSs[n].interference*1e6)
        return np.array(obs)

    def reset(self,):
        self.k=0
        self.done=False
        reset_env()
        self.BSs= [Basestation(i) for i in range(para.N)]

        # for n in range(para.N):
        #     self.BSs[n].interference=0
        #     self.BSs[n].UDs=[]
        obs=self.get_all_obs(self.k)
        self.ACCs=np.zeros(para.K)
        self.final_ps=np.zeros(para.K)
        return obs
    def get_pmin(self,SCR,bs_id):
        lar_fad = self.BSs[bs_id].large_h[self.k]
        delta = 1
        interfence_k=self.BSs[bs_id].interference+para.N0*self.BSs[bs_id].W
        phy_k = para.d0 / (self.BSs[bs_id].W * para.t_max)
        pmin=interfence_k*(2**(phy_k*SCR)-1)/(lar_fad*delta*np.sqrt(-2*np.log(para.Prob_th)))
        return pmin


    def deal_each_bs(self,bs_id,SCR):
        # self.BSs[bs_id].user_set.append(self.k)
        pmin=self.get_pmin(SCR,bs_id)
        power=pmin
        flag=1
        if pmin>para.p_max[self.k]:
            self.done=True
            flag=0
            pass
        self.BSs[bs_id].UDs.append([int(self.k),int(10*SCR),pmin])

        # lar_fad = self.BSs[bs_id].large_h[self.k]
        sinr = power* self.BSs[bs_id].H[self.k] / (self.BSs[bs_id].interference + para.N0*self.BSs[bs_id].W)
        sinr_dB = 10 * np.log10(sinr)
        td=para.d0/(self.BSs[bs_id].W*np.log2(1+sinr))
        # Succe_Prob=para.Succe_Prob(power,SCR,lar_fad,self.BSs[bs_id].W,self.BSs[bs_id].interference)
        self.BSs[bs_id].interference +=power*self.BSs[bs_id].large_h[self.k]*self.BSs[bs_id].small_h[self.k]
        # 做拟合了
        scr_int=int(SCR*10)
        ACC=fitting.fitting_func2(scr_int, sinr_dB)
        if flag==0:
            ACC=0
        return ACC

    def step(self,action):
        #  一个大时隙里，模拟很多次用户请求 Reward设置为：成功1-时延；失败0
        # 模拟用户请求
        # 先确定选哪个BS
        BS_id = action[0]
        SCR=(action[1]+1)/10  #0.1，0.2,1
        reward=self.deal_each_bs(BS_id,SCR)
        self.k+=1
        # ACC_s = None
        if self.k==para.K:
            self.done=True
            next_obs=np.zeros(para.state_dim)

        else:
            next_obs=self.get_all_obs(self.k)
        if self.done:
            for n in range(para.N):
                self.BSs[n].UDs=np.array(self.BSs[n].UDs)
                if self.BSs[n].UDs.size==0:
                    continue
                ps,ACC_s=self.BSs[n].get_power()
                UDs= self.BSs[n].UDs[:,0].astype(int)
                for i,k in enumerate(UDs):
                    self.ACCs[k]=ACC_s[i]
                    self.final_ps[k]=ps[i]
        return next_obs, reward, self.done, self.ACCs


if __name__ == '__main__':
    b1=Basestation(1)

