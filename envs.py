# time: 2024/1/26 21:16
# author: YanJP
import numpy as np
import para

class Basestation():
    def __init__(self,id):
        self.id=id
        self.user_set=[]
        self.W=para.Ws[id]
        self.interference=0
        self.large_h=para.large_fading[id]
        self.small_h=para.h[id]

    def get_ini_obs(self,k):
        H=self.large_h[k]*self.small_h[k]
        self.obs=np.array([self.W/1e6,self.interference,self.large_h[k],self.small_h[k],H])
        # self.obs=np.concatenate((self.interference,self.large_h[k],self.small_h[k],H),axis=0)
        return self.obs
    def get_next_obs(self,action):
        if np.isin(action, self.video_cache):
            self.obs = np.concatenate((self.all_video_hot, self.dummy_video_cache), axis=0)
            # self.obs = self.dummy_video_cache
            # self.get_dummy()
            return self.obs  #下一个状态
        self.obs=np.concatenate((self.all_video_hot,self.dummy_video_cache),axis=0)
        return self.obs


class env_():
    def __init__(self):
        self.n=para.N
        self.action_space=para.action_dim
        self.observation_space=para.state_dim  #每个智能体的空间维度
        # self.UserAll=trans.generateU()
        self.reward=0
        self._max_episode_steps=para.K
        # self.h=para.h
        # self.min_simis=para.min_sims
        # self.salency=para.salency
        # self.request_tiles=np.random.randint(para.N_fov_low,para.N_fov)
        pass
    def get_all_obs(self,k):
        obs=[]
        for n in range(para.N):
            obs.append(self.BSs[n].get_ini_obs(k))
        return np.array(obs)

    def reset(self,):
        self.BSs= [Basestation(i) for i in range(para.N)]
        self.done=[0]*para.N
        self.k=0
        obs=self.get_all_obs(self.k)
        return obs

    def deal_each_bs(self,bs_id,power,SCR):
        self.BSs[bs_id].user_set.append(self.k)
        power*=para.p_max
        SCR=round(SCR*10)/10
        ACC=para.ACC*SCR
        lar_fad=self.BSs[bs_id].large_h[self.k]
        sinr=lar_fad* self.BSs[bs_id].small_h[self.k]/(self.BSs[bs_id].interference+para.sigma2)
        td=para.d0/(self.BSs[bs_id].W*np.log2(1+sinr))
        Succe_Prob=para.Succe_Prob(power,SCR,lar_fad,self.BSs[bs_id].W,self.BSs[bs_id].interference)
        self.BSs[bs_id].interference +=power*self.BSs[bs_id].large_h[self.k]*self.BSs[bs_id].small_h[self.k]
        reward=ACC*Succe_Prob
        return reward

    def step(self,action):
        #  一个大时隙里，模拟很多次用户请求 Reward设置为：成功1-时延；失败0
        # 模拟用户请求
        rewards=[]
        # 先确定选哪个BS
        first_column = action[:, 0]
        # max_value = np.max(first_column)
        max_BS= np.argmax(first_column) # 找到最大值所在的行索引
        reward=self.deal_each_bs(max_BS,action[max_BS,1],action[max_BS,2])
        rewards=[reward/para.N]*para.N
        self.k+=1
        next_obs=[]
        if self.k==para.K:
            self.done=[1.0,1.,1.]
        for i in range(para.N):
            each_obs=self.BSs[i].get_ini_obs(self.k)
            next_obs.append(each_obs)
        return next_obs, rewards, self.done, None



if __name__ == '__main__':
    b1=Basestation(1)

