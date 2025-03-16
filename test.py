# 1.初始化节点，为每个节点分配固定的活跃度，并且分配随机意见
# 2.计算节点之间的同质性
# 3.网络连接，并保存邻接矩阵
# 4.使用Bron-Kerbosch算法找出所有极大团，将其存储为单纯形列表
# 5.观点整合，单纯形内部进行意见交互
# 6.信息传播，单纯形之间进行意见交互
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import xgi
from models.test import HigherOrderOpinionDynamics
import os
from collections import defaultdict

# 图片保存位置
save_folder = "./plots"
if not os.path.exists(save_folder):
    print(f"创建文件夹 '{save_folder}' 。")
    os.makedirs(save_folder)  # 如果文件夹不存在，则创建
else:
    print(f"文件夹 '{save_folder}' 已存在。")



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体字体，SimHei 是常见的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class model():
    def __init__(self):
        pass

    def data_in(self, **kwargs):
        self.tips = kwargs.get("tips")
        self.N = kwargs.get("N")  # 代理数量
        self.T = kwargs.get("T")  # 时间步长
        self.alpha = kwargs.get("alpha")
        self.beta = kwargs.get("beta")
        self.gamma = kwargs.get("gamma")
        self.epsilon = kwargs.get("epsilon")
        self.m = kwargs.get("m")
        self.r = kwargs.get("r")
        self.dt = kwargs.get("dt")
        self.gamma_d = kwargs.get("gamma_d")  # γ1, γ2, γ3

        self.states = []  # 0: 未感染, 1: 感染
        self.activities = self.activities_get()
        self.A = np.zeros((self.N, self.N))  # 邻接矩阵
        self.S =[]
        self.infection_rates = []  # 记录每个时间步的感染率
        pass

    def activities_get(self):
        temp = np.random.uniform(0, 1, self.N)
        return (self.epsilon ** (1 - self.gamma) + temp * (1 - self.epsilon ** (1 - self.gamma))) ** (1 / (1 - self.gamma))

    def homogeneity_get(self, opinions):
        dif = np.abs(opinions[:, np.newaxis] - opinions)
        prob_matrix = (dif + 1e-10) ** (-self.beta)
        np.fill_diagonal(prob_matrix, 0)
        p_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        return p_matrix
    
    def network_update(self, tick):
        # 计算当前时间步的同质性矩阵 (仅一次)
        homogeneities = self.homogeneity_get(self.opinions[tick - 1])
        # print("homogeneities", homogeneities)

        # 向量化激活判断 (一次性生成所有激活节点)
        activated_mask = np.random.rand(self.N) <= self.activities
        activated_indices = np.where(activated_mask)[0]

        # 预分配所有需要的随机数 (激活节点数 * m)
        total_activated = len(activated_indices)

        # 初始化 self.S 用于存储符合条件的连接
        self.S = []

        # if total_activated > 0:
        #     # 批量生成所有邻居选择
        #     all_neighbors = np.array([
        #         np.random.choice(self.N, size=self.m, replace=False, p=homogeneities[i]) 
        #     ])

        #     # 过滤同质性大于 r 的邻居并存入 self.S
        #     for i, neighbors in zip(activated_indices, all_neighbors):
        #         # 取出当前节点 i 与其邻居的同质性
        #         neighbor_homogeneity = homogeneities[i][neighbors]
                
        #         # 筛选出同质性大于 r 的邻居
        #         valid_neighbors = neighbors[neighbor_homogeneity > self.r] # np.random.rand()
                
        #         # 如果有符合条件的邻居，则存入 self.S
        #         if len(valid_neighbors) > 0:
        #             self.S.append([i] + valid_neighbors.tolist())



        if total_activated > 0:
            # 批量生成所有邻居选择
            all_neighbors = np.array([
                np.random.choice(self.N, size=self.m, replace=False, p=homogeneities[i]) 
                for i in activated_indices
            ])

            # 存入 self.S
            for i, neighbors in zip(activated_indices, all_neighbors):
                self.S.append([i] + neighbors.tolist())

    def sis_transmission(self, simplex_list):
        """在单纯形结构中执行SIS传播"""
        # 恢复过程
        recovery_mask = np.random.rand(self.N) < self.mu
        self.states[recovery_mask] = 0
        
        # 构建感染影响力图
        infection_pressure = defaultdict(int)
        for simplex in simplex_list:
            infected_members = [n for n in simplex if self.states[n] == 1]
            for node in simplex:
                if self.states[node] == 0:  # 只影响易感节点
                    infection_pressure[node] += len(infected_members)
        
        # 计算感染概率
        for node, pressure in infection_pressure.items():
            infection_prob = 1 - (1 - self.beta)**pressure
            if np.random.rand() < infection_prob:
                self.states[node] = 1

    def simulate_SIS(self):
        self.states = np.zeros(self.N)# 初始化
        # 主循环
        for tick in tqdm(range(1, self.T)):
            # if tick > 1:
            #     break
            self.A = np.zeros((self.N, self.N))  # 重置邻接矩阵
            self.S =[]  # 重置单纯形列表
            self.network_update(tick)# 网络连接 
            self.sis_transmission(self, self.S)# 传播
            # 记录感染率
            infection_rate = self.states.mean()
            self.infection_rates.append(infection_rate)

    
    def analyze_threshold(self, beta_range, mu_fixed=0.1, trials=5):
        """分析不同beta值下的传播阈值"""
        steady_states = []
        
        for beta in tqdm(beta_range):
            self.beta = beta
            self.mu = mu_fixed
            trial_rates = []
            
            for _ in range(trials):
                self.states = np.zeros(self.N)# 初始化
                self.simulate_SIS()
                # 取最后100步的平均作为稳态值
                trial_rates.append(np.mean(self.infection_rates[-100:]))
                
            steady_states.append(np.mean(trial_rates))
        
        # 找到阈值转折点
        threshold_index = np.argmax(np.diff(steady_states) > 0.01)
        threshold_beta = beta_range[threshold_index] if threshold_index > 0 else None
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(beta_range, steady_states, 'o-', color='darkred')
        if threshold_beta:
            plt.axvline(threshold_beta, color='k', linestyle='--', 
                      label=f'估计阈值: {threshold_beta:.3f}')
        plt.xlabel('感染率 (β)')
        plt.ylabel('稳态感染比例')
        plt.title('SIS传播阈值分析')
        plt.legend()
        plt.grid(True)
        self.save()
        plt.show()
            
    def save(self):
        global save_folder
        # 保存图像
        file_prefix = "Fig"  # 文件名前缀
        file_extension = ".svg"  # 文件扩展名
        
        # 获取文件夹中已存在的文件数量
        existing_files = [f for f in os.listdir(save_folder) if f.startswith(file_prefix) and f.endswith(file_extension)]
        
        # 找到下一个可用的编号
        next_number = 1
        while True:
            file_name = f"{file_prefix}_{next_number}{file_extension}"
            save_path = os.path.join(save_folder, file_name)
            if not os.path.exists(save_path):
                break
            next_number += 1

        # 保存图像
        plt.savefig(save_path, format='svg', dpi=100)
        print(f"图像已保存至: {save_path}")



if __name__ == '__main__':
    model = model()


    # 配置参数
    config = {
        "tips": "加入高阶交互",
        "N": 100,  # 代理数量
        "T": 1000,  # 时间步长
        "dt": 0.01,  # 时间步长
        "alpha": 0.05,  # 意见动态方程中的参数
        "beta": 2,  # 控制代理人选择互动对象的概率
        "K": 3,  # 意见动态方程中的参数
        "gamma": 2.1,  # 活动值分布的幂律指数
        "epsilon": 0.01,  # 活动值的最小值
        "m": 10,  # 每个活跃代理的连接数
        "r": 0.001,  # 交互阈值
        "gamma_d": [1.0, 0.5, 0.2]  # γ1, γ2, γ3
    }
    # 初始化模型

    model.data_in(**config)

    # 阈值分析（可能需要较长时间）
    beta_range = np.linspace(0.01, 0.5, 5)
    model.analyze_threshold(beta_range, mu_fixed=0.1, trials=3)