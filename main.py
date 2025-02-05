# import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import networkx as nx
import os
from src.module import tech
from src.module import GraphAnalyzer
from src.func import func
import xgi

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体字体，SimHei 是常见的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class model():
    def __init__(self):
        pass
    
    def data_in(self, **kwargs):
        # 参数
        self.N = kwargs.get("N")
        self.T = kwargs.get("T")
        self.dt = kwargs.get("dt")
        self.alpha = kwargs.get("alpha")
        self.beta = kwargs.get("beta")
        self.K = kwargs.get("K")
        self.gamma = kwargs.get("gamma")
        self.epsilon = kwargs.get("epsilon")
        self.m = kwargs.get("m")
        self.r = kwargs.get("r")

        self.activities = self.activities_get()
        self.A = np.zeros((self.N, self.N))

    def activities_get(self):
        temp = np.random.uniform(0, 1, self.N)
        return (self.epsilon ** (1 - self.gamma) + temp * (1 - self.epsilon ** (1 - self.gamma))) ** (1 / (1 - self.gamma))

    def homogeneity_get(self, opinions, node):
        p_matrix = np.zeros((self.N, self.N))  # 概率矩阵，表示代理人之间互动的概率

        for i in range(self.N):
            dif = np.abs(opinions[i] - opinions)  # 计算代理人 i 与其他代理人之间的意见距离
            prob = (dif + 1e-10) ** (-self.beta)  # 根据意见距离计算互动概率
            prob[i] = 0  # 自己与自己不互动
            p_matrix[i, :] = prob / np.sum(prob)  # 归一化概率
        return p_matrix
    
    def network_update(self, tick):
        for i in range(self.N):# 遍历所有节点，确定是否激活
            if np.random.rand() <= self.activities[i]:
                # 激活
                homogeneities = self.homogeneity_get(self.opinions[tick - 1], i)# 获取同质性
                neighbors = np.random.choice(self.N, size=self.m, replace=False, p=homogeneities[i])  # 选择m个节点连接
                # 连接
                self.A[i, neighbors] = 1
                # 互惠
                for node in neighbors:
                    if np.random.rand() < self.r:  # 以概率 r 设置反向关系
                        self.A[node][i] = 1
        # 网络显示
        # cliques = tech.bron_kerbosch_pivot(self.A)
        # print(cliques)
        # finder = MaximalCliqueFinder(self.A)
        # maximal_cliques = finder.find_cliques()
        

        #analyzer = GraphAnalyzer(self.A, directed=True)
        #maximal_cliques = analyzer.find_maximal_cliques()
        #print(self.A)
        #print(maximal_cliques)
        #func.network_print(self.A)
        
        #func.simplex_print(maximal_cliques)

    def opinion_dynamics(self, x):# 意见动态微分方程
        return -x + self.K * np.sum(self.A * np.tanh(self.alpha * x), axis=1)

    def runge_kutta(self, opinions):
        k1 = self.dt * self.opinion_dynamics(opinions)  # 计算 k1
        k2 = self.dt * self.opinion_dynamics(opinions + 0.5 * k1)  # 计算 k2
        k3 = self.dt * self.opinion_dynamics(opinions + 0.5 * k2)  # 计算 k3
        k4 = self.dt * self.opinion_dynamics(opinions + k3)  # 计算 k4
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 更新意见值


    def simulate_opinion_dynamics(self):# 意见动态模型
        self.x = np.random.uniform(-1, 1, self.N)# 初始化意见，范围为[-1, 1]
        self.opinions = np.zeros((self.T, self.N))  # 存储每个时间步的意见
        self.opinions[0] = self.x  # 初始意见
        # 主循环
        for tick in tqdm(range(1, self.T)):
            self.A = np.zeros((self.N, self.N))  # 重置邻接矩阵
            self.network_update(tick)# 网络连接
            opinions_temp = self.runge_kutta(self.opinions[tick - 1])# 意见更新
            #print(opinions_temp)
            self.opinions[tick] = self.opinions[tick - 1] + opinions_temp  # 记录当前时间步的意见
            

    # def save(self):
    #     global save_folder
    #     # 保存图像
    #     file_prefix = "Fig"  # 文件名前缀
    #     file_extension = ".png"  # 文件扩展名
    #     # 获取文件夹中已存在的文件数量
    #     existing_files = [f for f in os.listdir(save_folder) if f.startswith(file_prefix) and f.endswith(file_extension)]
    #     next_number = len(existing_files) + 1  # 下一个编号

    #     # 生成文件名
    #     file_name = f"{file_prefix}_{next_number}{file_extension}"
    #     save_path = os.path.join(save_folder, file_name)

    #     # 保存图像
    #     plt.savefig(save_path)
    #     print(f"图像已保存至: {save_path}")

    

if __name__ == '__main__':
    model = model()
    func = func(model)
    tech = tech()
    # 定义矩阵存放数据
    lengh = 1
    
    

    # 配置参数
    # 中立0.05, 2
    # 激进化3, 0
    # 极化3, 3
    config = {
        "N": 100,  # 代理数量
        "T": 1000,  # 时间步长
        "dt": 0.01,  # 时间步长
        "alpha": 3,  # 意见动态方程中的参数
        "beta": 0,  # 控制代理人选择互动对象的概率
        "K": 3,  # 意见动态方程中的参数
        "gamma": 2.1,  # 活动值分布的幂律指数
        "epsilon": 0.01,  # 活动值的最小值
        "m": 10,  # 每个活跃代理的连接数
        "r": 0.5,  # 互动的互惠性参数
    }

    func.opinions_draw(config)# 绘制 opinion 图
    # func.heatmap(lengh, config)# 绘制热力图

