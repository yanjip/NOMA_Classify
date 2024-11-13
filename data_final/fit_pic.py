import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取上传的 Excel 文件
file_path = './output.xlsx'
data = pd.read_excel(file_path)


# 定义拟合函数（假设为Logistic函数）
def logistic_function(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


# 重新定义SNR范围为0到20，每隔1一个点
snr_values = np.arange(-10, 11, 1)

# 创建一个图形对象
# plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font', size=15)

# 遍历每列数据并进行拟合
fit_params=np.zeros((10,3))
i=-1
cols=['test_acc_2','test_acc_4','test_acc_6','test_acc_8','test_acc_10']
# for column in data.columns:
for column in cols:

    # if (i+1)%2==1:
    #     i += 1
    #     continue
    accuracy = data[column].values[:len(snr_values)]  # 只取前21个数据点，以匹配SNR长度

    # 拟合曲线
    popt, _ = curve_fit(logistic_function, snr_values, accuracy, maxfev=10000)
    print(f'Fitted parameters for {column}: a = {popt[0]:.4f}, b = {popt[1]:.4f}, c = {popt[2]:.4f}')
    fit_params[i]=np.array([popt[0],popt[1],popt[2]])
    i+=1
    # 使用拟合参数计算拟合曲线
    snr_fine = np.linspace(-10, 11, 200)
    fitted_curve = logistic_function(snr_fine, *popt)

    # 绘制原始数据点和拟合曲线
    # plt.plot(snr_values, accuracy, 'o', label=f'{column} Data')
    if column=='test_acc_10':
        plt.plot(snr_fine, fitted_curve, '-', label=f'SCR=1.0')
    else:
        plt.plot(snr_fine, fitted_curve, '-', label=f'SCR=0.{column[-1]}')


# np.save('fit_params.npy', fit_params)
print(fit_params)
# 添加图形标签和标题
plt.xlabel('SNR (dB)',labelpad=-1)
plt.ylabel('Classification Accuracy',labelpad=-1)
plt.legend(ncol=1)
plt.grid(linestyle=":", color="gray", linewidth="0.01", axis="both")
# plt.savefig('fit_result.png',dpi=600)
plt.savefig("pic/fit_"  , dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.savefig("pic/fit_"  +'.eps',format='eps', dpi=600, bbox_inches='tight', pad_inches=0.01)

# 显示图形
plt.show()