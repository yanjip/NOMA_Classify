# time: 2024/10/24 21:46
# author: YanJP
import numpy as  np
import para
# 定义p和H的数组
np.random.seed(96)
p = np.array([1]*para.K)  # 替换成实际的p值
H = np.random.random(para.K)*2 # 替换成实际的H值

# 定义常量N_0和W_n
N_0 = 1.00  # 根据实际情况替换
W_n = 1  # 根据实际情况替换
N0_Wn=N_0*W_n
lower_bounds=[0.2]*(len(p))
upper_bounds=np.random.random(para.K)*8+0.2


# 定义投影函数，将p1, p2, p3投影到给定上下界内
def project_onto_bounds(p, lower_bounds, upper_bounds):
    return np.clip(p, lower_bounds, upper_bounds)

def get_each_k_gradient(p,k,its):
    gamma_k=p[k]*H[k]/(np.sum(p[:k] * H[:k]) + N_0 * W_n)
    common_part=1*np.exp(-gamma_k-1)/(1+np.exp(-gamma_k-1))**2
    Hk_copy=H[k]
    fenmu =  (np.sum(p[:k] * H[:k]) + N_0 * W_n)
    grad=Hk_copy/fenmu  # 计算第一项
    for i in range(its):
        fenzi=p[k+1]*H[k+1]*Hk_copy
        fenmu=fenmu+p[k]*H[k]
        grad=grad-fenzi/(fenmu**2)
        k+=1
    return common_part*grad


def gradient(ps):
    df_powers=np.zeros(para.K)
    for k in range(para.K-1,-1,-1):
        ans=get_each_k_gradient(ps, k,para.K-1-k)
        df_powers[k]=ans
    return np.array(df_powers)
def get_gamma_k(p,H,k):
    gamma_k=p[k]*H[k]/(np.sum(p[:k] * H[:k]) + N_0 * W_n)
    return gamma_k

def objective(p):
    acc=0
    for k in range( para.K):
        gamma_k=get_gamma_k(p,H,k)
        sinr_dB = 10 * np.log10(gamma_k)
        acc+=1/(np.exp(-1-0.5*sinr_dB))
    return acc

# 梯度投影算法
def gradient_projection_algorithm(grad, p0, alpha=0.2, iterations=1200):
    p = p0
    for i in range(iterations):
        # 计算梯度
        grad_values = grad(p)

        # 梯度上升步（因为要最大化）
        grad_step = p + alpha * grad_values

        # 投影到约束范围内
        p = project_onto_bounds(grad_step, lower_bounds, upper_bounds)
        # gamma_ks = []
        # for k in range(len(p)):
        #     gamma_k = get_gamma_k(p, H, k)
        #     gamma_ks.append(gamma_k)
        # print("gamma_ks:", gamma_ks)
        # 输出每次迭代的结果
        print(f"Iteration {i + 1}: p = {p}, f(p) = {objective(p)}")
    print("upper:", upper_bounds)
    p[-1]=upper_bounds[-1]
    print(f"Iteration -1: p = {p}, f(p) = {objective(p)}")


    return p

if __name__ == '__main__':
    # 初始值
    p0 = lower_bounds # 初始功率值
    # p0[-1]=upper_bounds[-1]
    # print("初始值：", p0)
    # 运行梯度投影算法
    print(p,H)
    final_p = gradient_projection_algorithm(gradient, p0)
    # print(
    #     f"Final result: p1 = {final_p[0]}, p2 = {final_p[1]}, p3 = {final_p[2]}, f(p) = {objective(final_p[0], final_p[1], final_p[2])}")

    print("max allocation:")
    p=upper_bounds
    print(p)
    print(f" f(p) = {objective(p)}")



