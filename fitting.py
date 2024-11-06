# time: 2024/1/17 21:57
# author: YanJP
import numpy as np
import matplotlib.pyplot as plt
import para

compress=np.array([0.4,0.5,0.6,0.7,0.8])

def get_model(o):
    if o==0:
        return 0.6079,0.1233,0.4463,0.2523,0.353,0.2557,0.2197
    elif o==1:
        return 0.7198,0.08013,0.3929,0.2674,0.4787,0.2724,0.5449
    elif o==2:
        return  0.6639,0.3682,1.282,0.1845,0.269,0.1784,1.187
    elif o==3:
        return 0.8284,0.03301,0.2542,0.2244,1.055,0.2602,0.907
    elif o==4:
        return  0.8448,0.07156,0.6048,0.2308,0.3392,0.2432,0.7652
    elif o==5:
        return 0.8432,0.1361,1.082,0.2042,0.1193,0.9602,1.069
    elif o==6:
        return 0.8871,0.05013,0.5978,0.232,0.2959,0.7714,0.6039
    elif o==7:
        return 0.9154,0.002834,0.05225,0.29,2.09,0.389,1.293
    elif o==8:
        return 0.8925,0.05973,0.7253,0.1951,0.3607,0.3932,0.9652
    elif o==9:
        return 0.9122,0.05267,0.7994,0.1948,0.07356,0.983,0.5041


def fitting_func2(ratio,snr):
    a,b,c=para.fit_params[ratio-1]
    acc=c / (1 + np.exp(-a * (snr - b)))
    return acc


def fitting_func(q,r):
    a,b,c,d,e,m,n=get_model(q)
    fenmu=c+np.exp(-(d*r+e))+np.exp(-(m*np.power(r, 3)+n))
    sim=a + b / fenmu
    return sim
def pic():
    snr=np.arange(0,12,1)
    y0=fitting_func(0,snr)
    y1=fitting_func(1,snr)
    y2=fitting_func(2,snr)
    y3=fitting_func(3,snr)
    y4=fitting_func(4,snr)
    y5=fitting_func(5,snr)
    y6=fitting_func(6,snr)
    y7=fitting_func(7,snr)
    y8=fitting_func(8,snr)
    y9=fitting_func(9,snr)
    plt.plot(snr, y0, 'r', label='0.1')
    plt.plot(snr, y1, 'b',label='0.2' )
    plt.plot(snr, y2, 'g', label='0.3')
    plt.plot(snr, y3, 'y', label='0.4')
    plt.plot(snr, y4, 'k', label='0.5')
    plt.plot(snr, y5,  label='0.6')
    plt.plot(snr, y6, label='0.7' )
    plt.plot(snr, y7, label='0.8')
    plt.plot(snr, y8, label='0.9')
    plt.plot(snr, y9, label='1.0')
    # 显示网格
    plt.grid(True)
    plt.legend()
    # 显示图像
    # plt.savefig('fitting.png')

    plt.show()

def pic2():
    snr=np.arange(-10,30,1)
    fit_para=para.fit_params
    y=[]
    for i in range(10):
        y.append(fitting_func2(i,snr))
    for i in range(10):
        plt.plot(snr, y[i], label=str(i*0.1+0.1))
    # 显示网格
    plt.grid(True)
    plt.legend()
    # 显示图像
    # plt.savefig('fitting.png')

    plt.show()




# print(snr)
if __name__ == '__main__':
    # pic()
    pic2()