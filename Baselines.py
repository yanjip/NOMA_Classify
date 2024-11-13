# time: 2024/10/25 22:09
# author: YanJP
from envs import Basestation
import para
import numpy as np
import fitting
def get_probability(power,ratio,lar_fad,W):
    phy=para.d0/(W*para.t_max)
    # xi=power/(N0_dBm*W)
    xi=power*lar_fad/(para.N0*W)
    delta=1
    x = (2 ** (phy * ratio) - 1) / (xi * delta)
    prob = np.exp(-x ** 2 / 2)
    return prob


# Baseline 2: alpha固定一个BS；SCR选择最小的；功率最小；
class Baseline2:
    def __init__(self, ):
        self.BSs= [Basestation(i) for i in range(para.N)]
        self.Hks=para.H
        self.scr=0.1
        self.ACCs=np.zeros(para.K)
        # self.get_acc()
    def get_pmin(self,SCR,bs_id,k):
        lar_fad = self.BSs[bs_id].large_h[k]
        delta = 1
        interfence_k=self.BSs[bs_id].interference+para.N0*self.BSs[bs_id].W
        phy_k = para.d0 / (self.BSs[bs_id].W * para.t_max)
        pmin=interfence_k*(2**(phy_k*SCR)-1)/(lar_fad*delta*np.sqrt(-2*np.log(para.Prob_th)))
        return pmin
    def get_acc(self,):
        for k in range(para.K):
            # BS_id=int(np.argmax(self.Hks[:,k], axis=0))
            # BS_id=np.random.randint(0,para.N)
            BS_id =2
            pmin = self.get_pmin(self.scr, BS_id,k)
            if pmin > para.p_max[k]:
                #     self.scr-=0.1
                # if self.scr==0.0:
                continue
            power = pmin
            power=para.p_max[k]
            sinr = power * self.BSs[BS_id].H[k] / (self.BSs[BS_id].interference + para.N0 * self.BSs[BS_id].W)
            sinr_dB = 10 * np.log10(sinr)
            scr_int = int(self.scr * 10)
            pro=get_probability(power,self.scr,self.BSs[BS_id].large_h[k],self.BSs[BS_id].W)
            if pro<para.Prob_th:
                continue
            ACC = fitting.fitting_func2(scr_int, sinr_dB)
            self.ACCs[k]=ACC
            self.BSs[BS_id].interference +=power*self.BSs[BS_id].large_h[k]*self.BSs[BS_id].small_h[k]
        return self.ACCs.mean()

# Baseline 3: alpha按信道强度；SCR选择最大的；功率最小；
class Baseline3:
    def __init__(self, ):
        self.BSs= [Basestation(i) for i in range(para.N)]
        self.Hks=para.H
        self.scr=1.0
        self.ACCs=np.zeros(para.K)
        # self.get_acc()
    def get_pmin(self,SCR,bs_id,k):
        lar_fad = self.BSs[bs_id].large_h[k]
        delta = 1
        interfence_k=self.BSs[bs_id].interference+para.N0*self.BSs[bs_id].W
        phy_k = para.d0 / (self.BSs[bs_id].W * para.t_max)
        pmin=interfence_k*(2**(phy_k*SCR)-1)/(lar_fad*delta*np.sqrt(-2*np.log(para.Prob_th)))
        return pmin
    def get_acc(self,):
        for k in range(para.K):
            BS_id=int(np.argmax(self.Hks[:,k], axis=0))
            pmin = self.get_pmin(self.scr, BS_id,k)
            # power=para.p_max[k]
            if pmin > para.p_max[k]:
            #     self.scr-=0.1
            # if self.scr==0.0:
                continue
            power=pmin
            sinr = power * self.BSs[BS_id].H[k] / (self.BSs[BS_id].interference + para.N0 * self.BSs[BS_id].W)
            sinr_dB = 10 * np.log10(sinr)
            scr_int = int(self.scr * 10)
            pro = get_probability(power, self.scr, self.BSs[BS_id].large_h[k], self.BSs[BS_id].W)
            if pro < para.Prob_th:
                continue
            ACC = fitting.fitting_func2(scr_int, sinr_dB)
            self.ACCs[k]=ACC
            self.BSs[BS_id].interference +=power*self.BSs[BS_id].large_h[k]*self.BSs[BS_id].small_h[k]
        return self.ACCs.mean()

# Baseline 4: alpha按信道最大；SCR选择遍历；功率最大；
class Baseline4:
    def __init__(self, ):
        self.BSs= [Basestation(i) for i in range(para.N)]
        self.Hks=para.H
        # self.scr=np.random.choice(para.scr_range,para.K)
        self.scr=np.ones(para.K)
        self.ACCs=np.zeros(para.K)
        # self.get_acc()
    def get_pmin(self,SCR,bs_id,k):
        lar_fad = self.BSs[bs_id].large_h[k]
        delta = 1
        interfence_k=self.BSs[bs_id].interference+para.N0*self.BSs[bs_id].W
        phy_k = para.d0 / (self.BSs[bs_id].W * para.t_max)
        pmin=interfence_k*(2**(phy_k*SCR)-1)/(lar_fad*delta*np.sqrt(-2*np.log(para.Prob_th)))
        return pmin
    def get_acc(self,):
        for k in range(para.K):
            BS_id=np.random.randint(0,para.N)
            # BS_id=int(np.argmax(self.Hks[:,k], axis=0))
            pmin = self.get_pmin(self.scr[k], BS_id,k)
            # power=para.p_max[k]
            if pmin > para.p_max[k]:
                continue
            # while pmin > para.p_max[k]:
            # if pmin > para.p_max[k]:
            #     self.scr[k]-=0.1
            #     pmin = self.get_pmin(self.scr[k], BS_id,k)
            #
            # if self.scr[k]==0.0:
            #     continue
            # power=np.random.uniform(pmin, para.p_max[k])
            # power=para.p_max[k]
            power=pmin
            sinr = power * self.BSs[BS_id].H[k] / (self.BSs[BS_id].interference + para.N0 * self.BSs[BS_id].W)
            sinr_dB = 10 * np.log10(sinr)
            scr_int = int(self.scr[k] * 10)
            pro = get_probability(power, self.scr[k], self.BSs[BS_id].large_h[k], self.BSs[BS_id].W)
            if pro < para.Prob_th:
                continue
            ACC = fitting.fitting_func2(scr_int, sinr_dB)
            self.ACCs[k]=ACC
            self.BSs[BS_id].interference +=power*self.BSs[BS_id].large_h[k]*self.BSs[BS_id].small_h[k]
        return self.ACCs.mean()

if __name__ == '__main__':
    b2=Baseline2()
    ans_b2=b2.get_acc()
    print("Baseline2:",ans_b2)

    b3 = Baseline3()
    ans_b3 = b3.get_acc()
    print("Baseline3:", ans_b3)

    b4 = Baseline4()
    ans_b4 = b4.get_acc()
    print("Baseline4:", ans_b4)

