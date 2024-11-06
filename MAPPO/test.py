# time: 2024/11/2 19:10
# author: YanJP
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 生成三组随机数据
# data1 = np.random.normal(loc=0, scale=1, size=100)
# data2 = np.random.normal(loc=1, scale=1.5, size=100)
# data3 = np.random.normal(loc=-1, scale=0.5, size=100)
#
# # 将多组数据放入列表中
# data = [data1, data2, data3]
#
# # 绘制箱线图
# plt.boxplot(data, widths=0.05)
# plt.title('Box Plot of Multiple Data Sets')
# plt.xlabel('Data Sets')
# plt.ylabel('Values')
# plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 生成三组随机数据
data1 = np.random.normal(loc=0, scale=1, size=100)
data2 = np.random.normal(loc=1, scale=1.5, size=100)
data3 = np.random.normal(loc=-1, scale=0.5, size=100)

# 将多组数据放入列表中
data = [data1, data2, data3]

# 绘制仅显示均值和竖线的箱线图
plt.boxplot(data, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
            boxprops={"color": "None"},         # 隐藏箱体
            whiskerprops={"color": "black"},    # 显示竖线（须）
            capprops={"color": "black"},        # 显示竖线的端点
            flierprops={"color": "None"})       # 隐藏异常值
plt.title('Box Plot Showing Means and Vertical Lines Only')
plt.xlabel('Data Sets')
plt.ylabel('Values')
plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])
plt.show()

