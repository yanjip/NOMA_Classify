# time: 2024/10/24 15:52
# author: YanJP
import numpy as np

# 定义常数
H1, H2, H3 = 2.0, 1.5, 1.0  # 信道增益
N0 = 0.1  # 噪声功率
Wn = 2.0  # 带宽


# 定义目标函数 f
def objective(p1, p2, p3):
    term1 = p1 * H1 / (N0 * Wn)
    term2 = p2 * H2 / (p1 * H1 + N0 * Wn)
    term3 = p3 * H3 / (p1 * H1 + p2 * H2 + N0 * Wn)
    return 3 + term1 + term2 + term3


# 计算目标函数的梯度
def gradient(p1, p2, p3):
    # 对 p1 的偏导数
    df_dp1 = (H1 / (N0 * Wn)) - (p2 * H2 * H1) / ((p1 * H1 + N0 * Wn) ** 2) - (p3 * H3 * H1) / (
                (p1 * H1 + p2 * H2 + N0 * Wn) ** 2)

    # 对 p2 的偏导数
    df_dp2 = (H2 / (p1 * H1 + N0 * Wn)) - (p3 * H3 * H2) / ((p1 * H1 + p2 * H2 + N0 * Wn) ** 2)

    # 对 p3 的偏导数
    df_dp3 = H3 / (p1 * H1 + p2 * H2 + N0 * Wn)

    return np.array([df_dp1, df_dp2, df_dp3])


# 定义投影函数，将p1, p2, p3投影到给定上下界内
def project_onto_bounds(p, lower_bounds, upper_bounds):
    return np.clip(p, lower_bounds, upper_bounds)


# 梯度投影算法
def gradient_projection_algorithm(grad, p0, alpha=0.01, lower_bounds=[0, 0, 0], upper_bounds=[10, 10, 10],
                                  iterations=100):
    p = p0
    for i in range(iterations):
        # 计算梯度
        grad_values = grad(p[0], p[1], p[2])

        # 梯度上升步（因为要最大化）
        grad_step = p + alpha * grad_values

        # 投影到约束范围内
        p = project_onto_bounds(grad_step, lower_bounds, upper_bounds)

        # 输出每次迭代的结果
        print(f"Iteration {i + 1}: p = {p}, f(p) = {objective(p[0], p[1], p[2])}")

    return p


# 初始值
p0 = np.array([1.0, 3.0, 1.0])  # 初始功率值
print("初始值：",p0)
# 运行梯度投影算法
final_p = gradient_projection_algorithm(gradient, p0)
print(
    f"Final result: p1 = {final_p[0]}, p2 = {final_p[1]}, p3 = {final_p[2]}, f(p) = {objective(final_p[0], final_p[1], final_p[2])}")

