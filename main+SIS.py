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
        # 网络参数
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

        self.activities = self.activities_get()
        self.A = np.zeros((self.N, self.N))  # 邻接矩阵
        self.S =[]
        # 传染参数
        self.B = kwargs.get("B")  # 信息传播强度
        self.mu = kwargs.get("mu")  # 恢复强度
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

        if total_activated > 0:
            # 批量生成所有邻居选择
            all_neighbors = np.array([
                np.random.choice(self.N, size=self.m, replace=False, p=homogeneities[i]) 
                for i in activated_indices
            ])

            # 存入 self.S
            for i, neighbors in zip(activated_indices, all_neighbors):
                self.S.append([i] + neighbors.tolist())
    
    def opinion_dynamics(self, x):
        # 意见自衰减
        temp = -x
        # 示例邻接矩阵和参数
        self.gamma_d = [1.0, 0.5, 0.2]  # γ1, γ2, γ3
        tech = HigherOrderOpinionDynamics(self.S, self.gamma_d, self.N ,max_order=3)
        # time.sleep(10)
        # print(x)
        # print(temp)
        dx = tech.dx_dt(x, temp, self.alpha)
        # print("相位变化率 dθ/dt:", dtheta)
        return dx

    def runge_kutta(self, opinions):
        k1 = self.dt * self.opinion_dynamics(opinions)  # 计算 k1
        k2 = self.dt * self.opinion_dynamics(opinions + 0.5 * k1)  # 计算 k2
        k3 = self.dt * self.opinion_dynamics(opinions + 0.5 * k2)  # 计算 k3
        k4 = self.dt * self.opinion_dynamics(opinions + k3)  # 计算 k4
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6  # 更新意见值

    def simulate_opinion_dynamics(self):
        self.x = np.random.uniform(-1, 1, self.N)# 初始化意见，范围为[-1, 1]
        self.opinions = np.zeros((self.T, self.N))  # 存储每个时间步的意见
        self.opinions[0] = self.x  # 初始意见
        SIS.initialize_status()# 初始化感染人群
        # 主循环
        for tick in tqdm(range(1, self.T)):
            # if tick > 1:
            #     break
            self.A = np.zeros((self.N, self.N))  # 重置邻接矩阵
            self.S =[]  # 重置单纯形列表
            self.network_update(tick)# 网络连接 
            opinions_temp = self.runge_kutta(self.opinions[tick - 1])# 意见更新
            self.opinions[tick] = self.opinions[tick - 1] + opinions_temp  # 记录当前时间步的意见
            SIS.states = SIS.spread_infection(self.S, SIS.states, self.B, self.mu)
            # 记录感染率
            infection_rate = SIS.states.mean()
            SIS.infection_rates.append(infection_rate)

class SIS:
    def __init__(self):
        self.infection_rates = []

    def initialize_status(self):
        self.states = np.zeros(model.N)  # 0: 未感染, 1: 感染
        # 随机设置小部分感染者
        num_infected = int(model.N * 0.001)  # 假设初始感染人数为总人数的1%
        infected_indices = np.random.choice(model.N, size=num_infected, replace=False)
        self.states[infected_indices] = 1

    def spread_infection(self, S, states, beta, mu):
        """
        SAD模型下的SIS疾病传播
        参数:
        - S: 单纯形列表，每个单纯形是一个节点编号的集合
        - states: 节点状态列表 (0: 易感者, 1: 感染者)
        - beta: 感染率
        - mu: 恢复率

        返回:
        - 更新后的节点状态列表
        """
        self.beta = beta
        self.mu = mu
        new_states = states.copy()

        # 遍历每个单纯形，检查感染传播
        for simplex in S:
            infected = [node for node in simplex if states[node] == 1]
            susceptible = [node for node in simplex if states[node] == 0]

            if infected:
                for node in susceptible:
                    if np.random.rand() < self.beta:
                        new_states[node] = 1  # 自激活机制（已存在）

            # 他人激活机制（新增）
            if susceptible:
                for node in susceptible:
                    if any(states[n] == 1 for n in simplex):
                        if np.random.rand() < self.beta:
                            new_states[node] = 1

        # 每个感染者以概率 mu 恢复为易感者
        for i in range(len(states)):
            if states[i] == 1 and np.random.rand() < self.mu:
                new_states[i] = 0

        return new_states

class view():
    def __init__(self):
        pass

    def opinions_draw(self):
        model.simulate_opinion_dynamics()
        #self.draw()
        plt.figure(figsize=(10, 6))
        for i in range(model.N):
            plt.plot(range(model.T), model.opinions[:, i], alpha=0.5, linewidth=0.5)  # 绘制每个代理的意见随时间变化
        plt.xlabel('时间')
        plt.ylabel('意见')
        plt.title(f'{model.tips}  N={model.N},alpha={model.alpha},beta={model.beta},r={model.r},[γ1, γ2, γ3]={model.gamma_d}')
        self.save()
        plt.show(block=False)

    def heatmap(self, lengh, n, config, b, min_b, max_b, a, min_a, max_a):
        """
        绘制热力图
        :param lengh: 热力图的精细度
        :param n: 运行次数
        :param config: 配置参数
        :param min_b: K的最小值
        :param max_b: K的最大值
        :param min_a: alpha的最小值
        :param max_a: alpha的最大值
        """
        matrix = np.zeros((lengh, lengh))
        config[a] = min_a
        config[b] = min_b
        
        # 横向扫描
        for i in range(lengh):
            print(i)
            config[a] = 0
            for j in range(lengh):
                temp_average = [] 
                for _ in range(n):  # 多次运行取平均值
                    model.data_in(**config)
                    model.simulate_opinion_dynamics()
                    temp = np.abs(model.opinions[-1])
                    temp_average.append(np.average(temp))
                print(temp_average)
                matrix[i][j] = np.average(temp_average)
                print(f'{b}:{config[b]},{a}:{config[a]},结果{matrix[i][j]}')
                config[a] += (max_a - min_a) / (lengh)
            config[b] += (max_b - min_b) / (lengh)
        
        # 绘制热力图
        plt.imshow(matrix, cmap='gray', vmin=np.min(matrix), vmax=np.max(matrix), origin='lower')
        plt.colorbar()
        
        # 自定义横纵坐标刻度
        x_ticks = np.linspace(min_b, max_b, lengh)  # K 的范围
        y_ticks = np.linspace(min_a, max_a, lengh)  # alpha 的范围
        plt.xticks(np.arange(lengh), labels=np.round(x_ticks, 2))  # 设置 X 轴刻度和标签
        plt.yticks(np.arange(lengh), labels=np.round(y_ticks, 2))  # 设置 Y 轴刻度和标签
        
        # 添加坐标轴标签
        plt.xlabel('K')
        plt.ylabel('alpha')
        plt.title(f"{b}:{min_b}->{max_b},{a}:{min_a}->{max_a},{n}次取平均值")
        
        # 保存和显示图像
        self.save()
        plt.show(block=False)

    def analyze_threshold(self, beta_range, mu_fixed=0.1, trials=5):
        """
        分析不同beta值下的传播阈值
        :param beta_range: beta值范围
        :param mu_fixed: 恢复率
        :param trials: 每个beta值的试验次数
        """
        steady_states = []
        
        for beta in tqdm(beta_range):
            model.beta = beta
            model.mu = mu_fixed
            trial_rates = []
            
            for _ in range(trials):
                model.simulate_opinion_dynamics()
                # 取最后100步的平均作为稳态值
                trial_rates.append(np.mean(SIS.infection_rates[-100:]))
                
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

    def infection_rates_draw(self, beta, mu_fixed):
        model.beta = beta
        model.mu = mu_fixed
        model.simulate_opinion_dynamics()
        # # 绘制感染率随时间的变化
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(model.T), SIS.infection_rates, color='darkred')


        positions = list(range(len(SIS.infection_rates)))

        # 创建画布
        plt.figure(figsize=(12, 6))

        # 绘制折线图
        plt.plot(positions, SIS.infection_rates, 
                marker='o',    # 显示数据点
                linestyle='-', # 连线样式
                linewidth=1,   # 连线粗细
                color='steelblue', 
                markersize=4)

        # 添加标签和标题
        plt.xlabel('Data Position', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Data Value Visualization by Position', fontsize=14)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 自动调整布局
        plt.tight_layout()

        # 显示图表
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
    view = view()
    SIS = SIS()


    # 配置参数
    config = {
        "tips": "加入高阶交互",
        "N": 1000,  # 代理数量
        "T": 1000,  # 时间步长
        "dt": 0.01,  # 时间步长
        "alpha": 0.05,  # 意见动态方程中的参数
        "beta": 2,  # 控制代理人选择互动对象的概率
        "K": 3,  # 意见动态方程中的参数
        "gamma": 2.1,  # 活动值分布的幂律指数
        "epsilon": 0.01,  # 活动值的最小值
        "m": 10,  # 每个活跃代理的连接数
        "r": 0.001,  # 交互阈值
        "gamma_d": [1.0, 0.5, 0.2],  # γ1, γ2, γ3
        "B": 0.3,
        "mu": 0.1
    }
    model.data_in(**config)
    view.opinions_draw()

    config["alpha"], config["beta"] = 3, 0

    model.data_in(**config)
    view.opinions_draw()

    # config["alpha"], config["beta"] = 3, 3

    # model.data_in(**config)
    # view.opinions_draw()

    # config["r"], config["beta"] = 0, 0.5
    # model.data_in(**config)
    # view.heatmap(10, 5, config, "r", 0, 0.01, "alpha", 0, 1)# 绘制热力图

    # config["r"], config["beta"] = 0, 0.5
    # model.data_in(**config)
    # view.heatmap(10, 5, config, "r", 0, 0.05, "alpha", 0, 2)

    #view.infection_rates_draw(0.00000001, 0.8)

    # beta_range = np.linspace(0, 1, 50)
    # view.analyze_threshold(beta_range, mu_fixed=0.1, trials=10)