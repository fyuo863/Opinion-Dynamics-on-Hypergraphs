import numpy as np
import matplotlib.pyplot as plt

# 参数设置
N = 2000  # 总节点数
T = 10    # 总时间步
s_values = np.arange(2, 10)  # 研究不同的单纯形大小 s
a_mean = 0.01  # 平均活动率
a_std = 0.005  # 活动率标准差

# 计算平均度
avg_degree_SAD = (N - 1) * (1 - np.exp(- T * (s_values - 1)**2 * a_mean / (N - 1)))

# 绘图
plt.figure(figsize=(6,4))
plt.plot(s_values, avg_degree_SAD, marker='o', label="SAD Model")
plt.xlabel("Simplex size s")
plt.ylabel("Average degree ⟨k_T⟩")
plt.title("SAD Model: Average degree vs Simplex size")
plt.legend()
plt.grid()
plt.show()
