# time: 2024/11/6 22:26
# author: YanJP
import para
from Draw_pic import *

def Prob_pic_single(power):
    lar_fad=para.large_fading[:,0]
    prob=[]
    for r in para.scr_range:
        prob_lar=[]
        for lar in lar_fad:
            prob_lar.append(para.get_probability(power,r,lar))
        prob.append(sum(prob_lar)/len(prob_lar))
    print(prob)
    return prob
def Prob_pic_Td():
    TDs=(np.array([0.08,0.10,0.12,0.14]))
    TDs=np.round(TDs,3)
    power=0.05
    res = np.zeros((len(TDs), len(para.scr_range)))
    for i, t in enumerate(TDs):
        para.t_max=t
        res[i, :] = Prob_pic_single(power)
    # np.save('runs/datas/prob_td.npy',res)
    TDs=np.array([f"{x:.2f}" for x in TDs])
    plot_Prob(TDs, res, label='Maximum delay',danwei='s')

def Prob_pic_W():
    Ws = (np.array([5, 5.5, 6, 6.5]) -2.5) * 1e6
    # TDs=np.round(TDs,3)
    power=0.0001
    res = np.zeros((len(Ws), len(para.scr_range)))
    for i, w in enumerate(Ws):
        para.W=w
        res[i, :] = Prob_pic_single(power)
    np.save('runs/datas/prob_bandwidth.npy',res)
    plot_Prob(Ws/1e6, res, label='Bandwidth', danwei='MHz')
if __name__ == '__main__':
    Prob_pic_W()
    # Prob_pic_Td()