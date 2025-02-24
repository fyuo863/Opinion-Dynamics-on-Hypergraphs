import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 参数设置
N = 1000          # 振子数量
K = 3          # 耦合强度
omega = np.random.normal(0, 1, N)  # 自然频率（高斯分布）
theta0 = np.random.uniform(0, 2*np.pi, N)  # 初始相位

# Kuramoto微分方程
def kuramoto(theta, t, N, K, omega):
    dtheta_dt = np.zeros(N)
    for i in range(N):
        sum_term = np.sum(np.sin(theta - theta[i]))  # 相位耦合项
        dtheta_dt[i] = omega[i] + (K/N) * sum_term
    return dtheta_dt

# 时间序列
t = np.linspace(0, 10, 1000)

# 数值求解
theta = odeint(kuramoto, theta0, t, args=(N, K, omega))

# 计算同步参数r(t)
r = np.abs(np.mean(np.exp(1j * theta), axis=1))

# 可视化
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(t, theta, alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Phase θ')

plt.subplot(122)
plt.plot(t, r)
plt.xlabel('Time')
plt.ylabel('Synchronization r(t)')
plt.ylim(0,1)
plt.tight_layout()
plt.show()