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
        self.tips = kwargs.get("tips")
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

    # def homogeneity_get(self, opinions):
    #     p_matrix = np.zeros((self.N, self.N))  # 概率矩阵，表示代理人之间互动的概率

    #     for i in range(self.N):
    #         dif = np.abs(opinions[i] - opinions)  # 计算代理人 i 与其他代理人之间的意见距离
    #         prob = (dif + 1e-10) ** (-self.beta)  # 根据意见距离计算互动概率
    #         prob[i] = 0  # 自己与自己不互动
    #         p_matrix[i, :] = prob / np.sum(prob)  # 归一化概率
    #     return p_matrix

    def homogeneity_get(self, opinions):
        dif = np.abs(opinions[:, np.newaxis] - opinions)
        prob_matrix = (dif + 1e-10) ** (-self.beta)
        np.fill_diagonal(prob_matrix, 0)
        p_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        return p_matrix
    
    def custom_operation(self, tick, i):# 自定义操作函数
        # 激活
        homogeneities = self.homogeneity_get(self.opinions[tick - 1])# 获取同质性
        neighbors = np.random.choice(self.N, size=self.m, replace=False, p=homogeneities[i])  # 选择m个节点连接
        # 连接
        self.A[i, neighbors] = 1
        # 互惠
        rand_arr1 = np.random.rand(self.m)
        reciprocal_nodes = neighbors[rand_arr1 < self.r]
        self.A[reciprocal_nodes, i] = 1

        

    def network_update(self, tick):
        for i in range(self.N):# 遍历所有节点，确定是否激活
            if np.random.rand() <= self.activities[i]:
                # 激活
                homogeneities = self.homogeneity_get(self.opinions[tick - 1])# 获取同质性
                neighbors = np.random.choice(self.N, size=self.m, replace=False, p=homogeneities[i])  # 选择m个节点连接
                # 连接
                self.A[i, neighbors] = 1
                # 互惠
                rand_arr1 = np.random.rand(self.m)
                reciprocal_nodes = neighbors[rand_arr1 < self.r]
                self.A[reciprocal_nodes, i] = 1


                for node in neighbors:
                    if np.random.rand() < self.r:  # 以概率 r 设置反向关系
                        self.A[node][i] = 1


        # 优化


        # rand_arr2 = np.random.rand(self.N)
        # # 比较 array1 的元素是否大于 array2 的对应元素
        # comparison = rand_arr2 <= self.activities
        # indices = np.where(comparison)[0]
        # # 使用 np.vectorize 来向量化 temp 函数
        # vectorized_temp = np.vectorize(self.custom_operation)
        # # 执行向量化的 temp 函数
        # #start_time = time.perf_counter()
        # vectorized_temp(tick, indices)
        # #end_time = time.perf_counter()
        # #print(f"函数执行时间: {end_time - start_time:.6f} 秒")

        



        # 网络显示
        # cliques = tech.bron_kerbosch_pivot(self.A)
        # print(cliques)
        # finder = MaximalCliqueFinder(self.A)
        # maximal_cliques = finder.find_cliques()
        

        analyzer = GraphAnalyzer(self.A, directed=True)
        self.maximal_cliques = analyzer.find_maximal_cliques(shwo=False)#显示孤立节点
        #print(self.A)
        # print(self.maximal_cliques,"🍎")
        # print("------------------")
        # # for item in range(self.N):
        # #     #print(len(tech.find_simplex_with_node(self.maximal_cliques, item)))
        # #     print(tech.find_simplex_with_node(self.maximal_cliques, item))
        
        # #func.network_print(self.A)
        
        # func.simplex_print(self.maximal_cliques)

    def opinion_dynamics1(self, x):# 意见动态微分方程
        temp = -x
        for item in range(self.N):
            
            simplex = tech.find_simplex_with_node(self.maximal_cliques, item)
            if len(simplex) > 0:
                # print("------")
                # print("🍌", item, temp[item])
                
                # print(simplex)
                # print("------")
                
                for j in simplex:# 用指定节点的意见加上超边中其他所有节点的意见
                    # print("与item相连的边",j)
                    sum_rest = 0
                    for k in j:
                        # print("j=",j)
                        if k != item:
                            # print("边中包含的节点",k)
                            sum_rest += x[k]
                            # print("节点的意见", x[item], x[k], sum_rest)

                    temp[item] += self.K * 2 * len(j) * np.tanh(self.alpha * (sum_rest))
                    # print("temp计算完🍍", temp[item])
        return temp
        # return -x + self.K * np.sum(self.A * np.tanh(self.alpha * x), axis=1)
    
    def opinion_dynamics234(self, x):# 意见动态微分方程
        temp = -x
        for item in range(self.N):
            
            simplex = tech.find_simplex_with_node(self.maximal_cliques, item)
            if len(simplex) > 0:
                for j in simplex:# 用指定节点的意见加上超边中其他所有节点的意见
                    sum_rest = 0
                    for k in j:
                        if k != item:
                            sum_rest += x[k]
                    temp[item] += self.K * 2 * len(j) * np.tanh(self.alpha * (sum_rest))

        return temp
        # return -x + self.K * np.sum(self.A * np.tanh(self.alpha * x), axis=1)

    def runge_kutta(self, opinions):
        k1 = self.dt * self.opinion_dynamics1(opinions)  # 计算 k1
        k2 = self.dt * self.opinion_dynamics234(opinions + 0.5 * k1)  # 计算 k2
        k3 = self.dt * self.opinion_dynamics234(opinions + 0.5 * k2)  # 计算 k3
        k4 = self.dt * self.opinion_dynamics234(opinions + k3)  # 计算 k4
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
            self.opinions[tick] = self.opinions[tick - 1] + opinions_temp  # 记录当前时间步的意见
            
    

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
        "tips": "",
        "N": 500,  # 代理数量
        "T": 1000,  # 时间步长
        "dt": 0.01,  # 时间步长
        "alpha": 0.05,  # 意见动态方程中的参数
        "beta": 2,  # 控制代理人选择互动对象的概率
        "K": 3,  # 意见动态方程中的参数
        "gamma": 2.1,  # 活动值分布的幂律指数
        "epsilon": 0.01,  # 活动值的最小值
        "m": 10,  # 每个活跃代理的连接数
        "r": 0.5,  # 互动的互惠性参数
    }

    func.opinions_draw(config)# 绘制 opinion 图

    config["alpha"], config["beta"] = 3, 0

    func.opinions_draw(config)# 绘制 opinion 图

    config["alpha"], config["beta"] = 3, 3

    func.opinions_draw(config)# 绘制 opinion 图

    # func.heatmap(lengh, config)# 绘制热力图

    # func.finish_draw()

