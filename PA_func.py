# time: 2024/10/24 21:46
# author: YanJP
import numpy as  np
import para
# 定义p和H的数组


# 定义常量N_0和W_n




# 定义投影函数，将p1, p2, p3投影到给定上下界内
def project_onto_bounds(p, lower_bounds, upper_bounds):
    return np.clip(p, lower_bounds, upper_bounds)

def get_each_k_gradient(p,k,its,H):
    gamma_k=p[k]*H[k]/(np.sum(p[:k] * H[:k]) + para.N0*para.W_n)
    a, b, c = para.fit_params[para.SCRs[k] - 1]
    gamma_k_db= 10 * np.log10(gamma_k)

    common_part=10/(gamma_k*np.log(10))*a*c*np.exp(-a*(gamma_k_db-b))/(1+np.exp(-a*(gamma_k_db-b)))**2
    Hk_copy=H[k]
    fenmu =  np.sum(p[:k] * H[:k]) + para.N0*para.W_n
    grad=Hk_copy/fenmu  # 计算第一项
    for i in range(its):
        fenzi=p[k+1]*H[k+1]*Hk_copy
        fenmu=fenmu+p[k]*H[k]
        grad=grad-fenzi/(fenmu**2)
        k+=1
    return common_part*grad


def gradient(ps,H):
    df_powers=np.zeros(len(ps))
    for k in range(len(ps)-1,-1,-1):
        ans=get_each_k_gradient(ps, k,len(ps)-1-k,H)
        df_powers[k]=ans
    return np.array(df_powers)
def get_gamma_k(p,H,k):
    gamma_k=p[k]*H[k]/(np.sum(p[:k] * H[:k]) + para.N0*para.W_n)
    return gamma_k

def objective(p,H,Seperate=False):
    if Seperate:
        acc = []
        for k in range(len(p)):
            a, b, c = para.fit_params[para.SCRs[k] - 1]
            gamma_k = get_gamma_k(p, H, k)
            sinr_dB = 10 * np.log10(gamma_k)
            acc.append(c / (1 + np.exp(-a * (sinr_dB - b))))
        return acc
    else:
        acc=0
        for k in range( len(p)):
            a, b, c = para.fit_params[para.SCRs[k] - 1]
            gamma_k=get_gamma_k(p,H,k)
            sinr_dB = 10 * np.log10(gamma_k)
            acc+=c/(1+np.exp(-a*(sinr_dB-b)))
        return acc

# 梯度投影算法
def gradient_projection_algorithm(grad, p0,   lower_bounds, upper_bounds,H,alpha=0.0010, iterations=2000):
    p = p0
    # print(f"Iteration 0: p = {p}, f(p) = {objective(p, H)}")
    fp=objective(p,H)
    p_old=None
    for i in range(iterations):
        # 计算梯度
        grad_values = grad(p,H)

        # 梯度上升步（因为要最大化）
        grad_step = p + alpha * grad_values
        # 投影到约束范围内
        p = project_onto_bounds(grad_step, lower_bounds, upper_bounds)
        # gamma_ks=[]
        # for k in range(len(p)):
        #     gamma_k = get_gamma_k(p, H, k)
        #     gamma_ks.append(gamma_k)
        # print("gamma_ks:",gamma_ks)
        new_fp=objective(p,H)
        # 输出每次迭代的结果
        if new_fp-fp<0.000001 or new_fp<fp:
            break
        fp=new_fp
        print(f"Iteration {i + 1}: p = {p}, f(p) = {fp}")


    # print("upper:", upper_bounds)
    # p[-1]=upper_bounds[-1]
    # print(f"Iteration -1: p = {p}, f(p) = {objective(p)}")
    if objective(p0,H)>new_fp:
        p=p0
    return p
def PA_algor(H, lower_bounds, upper_bounds):
    p0 = lower_bounds # 初始功率值
    # p0[-1]=upper_bounds[-1]
    # print("初始值：", p0)
    # 运行梯度投影算法
    final_p = gradient_projection_algorithm(gradient, p0, lower_bounds, upper_bounds,H)
    # print("max allocation:",upper_bounds)
    # print(f" under max Power f(p) = {objective(upper_bounds)}")
    return final_p,objective(final_p,H,Seperate=True)

if __name__ == '__main__':
    # 初始值
    H = para.H[0]
    para.SCRs=np.array([6]*para.K)
    lower_bounds = np.array([0.1]*para.K)
    upper_bounds = para.p_max

    para.N0=4e-21  # 根据实际情况替换
    para.W_n=5e6
    PA_algor(H, lower_bounds, upper_bounds)
    print("min_power:",lower_bounds)

    print("max power:",upper_bounds)

    ans=[]



