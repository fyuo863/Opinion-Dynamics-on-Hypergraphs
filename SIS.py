import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgi
import os
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SAD_SIS_Model:
    def __init__(self):
        self.save_folder = "./sis_plots"
        os.makedirs(self.save_folder, exist_ok=True)

    def configure(self, config):
        # 网络参数
        self.N = config["N"]
        self.T = config["T"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.m = config["m"]
        self.r = config["r"]
        
        # SIS参数
        self.beta = config["beta"]
        self.mu = config["mu"]
        self.init_infected = config["init_infected"]
        
        # 初始化
        self.activities = self.activities_get()
        self.states = np.zeros(self.N)  # 0: Susceptible, 1: Infected
        self.infection_prob = np.zeros(self.N)
        self.initialize_states()
        
        # 结果存储
        self.infection_rates = []
        self.simplex_counts = []

    def activities_get(self):
        temp = np.random.uniform(0, 1, self.N)
        return (self.epsilon ** (1 - self.gamma) + temp * (1 - self.epsilon ** (1 - self.gamma))) ** (1 / (1 - self.gamma))

    def initialize_states(self):
        # 随机选择初始感染者
        initial_infected = np.random.choice(self.N, size=int(self.N*self.init_infected), replace=False)
        self.states[initial_infected] = 1

    def homogeneity_matrix(self):
        """生成同质性矩阵（简化为随机矩阵用于示例）"""
        return np.random.rand(self.N, self.N)

    def update_network(self):
        """生成当前时间步的单纯形结构"""
        homogeneities = self.homogeneity_matrix()
        activated = np.random.rand(self.N) < self.activities
        activated_indices = np.where(activated)[0]
        
        simplex_list = []
        for i in activated_indices:
            neighbors = np.random.choice(self.N, size=self.m, 
                                       p=homogeneities[i]/homogeneities[i].sum())
            simplex = [i] + neighbors.tolist()
            simplex_list.append(simplex)
            
        return simplex_list

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

    def run_simulation(self):
        for _ in tqdm(range(self.T)):
            # 更新网络结构
            simplex_list = self.update_network()
            self.simplex_counts.append(len(simplex_list))
            
            # 执行SIS传播
            self.sis_transmission(simplex_list)
            
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
                self.initialize_states()
                self.run_simulation()
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
        plt.savefig(os.path.join(self.save_folder, 'threshold_analysis.png'))
        plt.show()

    def visualize_results(self):
        """可视化感染传播过程"""
        plt.figure(figsize=(12, 6))
        
        # 感染率曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.infection_rates, color='darkblue')
        plt.xlabel('时间步')
        plt.ylabel('感染比例')
        plt.title('感染传播动态')
        plt.ylim(0, 1)
        
        # 单纯形数量变化
        plt.subplot(1, 2, 2)
        plt.plot(self.simplex_counts, color='darkgreen')
        plt.xlabel('时间步')
        plt.ylabel('活跃单纯形数量')
        plt.title('网络结构动态')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, 'dynamics_comparison.png'))
        plt.show()

if __name__ == '__main__':
    # 参数配置
    config = {
        "N": 1000,       # 节点数量
        "T": 1000,        # 时间步长
        "gamma": 2.1,    # 活动度分布参数
        "epsilon": 0.01, # 最小活动度
        "m": 5,          # 每个激活节点的连接数
        "r": 0.001,      # 同质性阈值
        "beta": 0.05,    # 初始感染概率
        "mu": 0.1,       # 恢复概率
        "init_infected": 0.01  # 初始感染比例
    }
    
    # 初始化模型
    model = SAD_SIS_Model()
    model.configure(config)
    
    # 运行单次模拟
    # model.run_simulation()
    # model.visualize_results()
    
    # 阈值分析（可能需要较长时间）
    beta_range = np.linspace(0.01, 0.5, 5)
    model.analyze_threshold(beta_range, mu_fixed=0.1, trials=3)