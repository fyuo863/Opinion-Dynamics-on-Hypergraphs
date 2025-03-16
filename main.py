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

            # # 批量设置连接
            # rows = np.repeat(activated_indices, self.m)
            # cols = all_neighbors.flatten()
            # self.A[rows, cols] = 1

        # # G = nx.from_numpy_array(self.A)
        # # self.S = [sublist for sublist in list(nx.find_cliques(G)) if len(sublist) > 1]



        # print("S", self.S)
        # time.sleep(10)
        
        # 创建单纯复形对象
        # H = xgi.SimplicialComplex()
        # # 添加单纯形到复形中
        # print("S", self.S)
        # H.add_simplices_from(self.S)
        # # 绘制图形并显示节点标签
        # xgi.draw(H, with_node_labels=True)
        # plt.title("Simplicial Complex Visualization")
        # plt.show()


    
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
        # 主循环
        for tick in tqdm(range(1, self.T)):
            # if tick > 1:
            #     break
            self.A = np.zeros((self.N, self.N))  # 重置邻接矩阵
            self.S =[]  # 重置单纯形列表
            self.network_update(tick)# 网络连接 
            opinions_temp = self.runge_kutta(self.opinions[tick - 1])# 意见更新
            self.opinions[tick] = self.opinions[tick - 1] + opinions_temp  # 记录当前时间步的意见
            
    def opinions_draw(self):
        self.simulate_opinion_dynamics()
        #self.draw()
        plt.figure(figsize=(10, 6))
        for i in range(self.N):
            plt.plot(range(self.T), self.opinions[:, i], alpha=0.5, linewidth=0.5)  # 绘制每个代理的意见随时间变化
        plt.xlabel('时间')
        plt.ylabel('意见')
        plt.title(f'{self.tips}  N={self.N},alpha={self.alpha},beta={self.beta},r={self.r},[γ1, γ2, γ3]={self.gamma_d}')
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
                    self.data_in(**config)
                    self.simulate_opinion_dynamics()
                    temp = np.abs(self.opinions[-1])
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
        "gamma_d": [1.0, 0.5, 0.2]  # γ1, γ2, γ3
    }
    # model.data_in(**config)
    # model.opinions_draw()

    # config["alpha"], config["beta"] = 3, 0

    # model.data_in(**config)
    # model.opinions_draw()

    # config["alpha"], config["beta"] = 3, 3

    # model.data_in(**config)
    # model.opinions_draw()

    config["r"], config["beta"] = 0, 0.5
    model.data_in(**config)
    model.heatmap(10, 5, config, "r", 0, 0.01, "alpha", 0, 1)# 绘制热力图

    config["r"], config["beta"] = 0, 0.5
    model.data_in(**config)
    model.heatmap(10, 5, config, "r", 0, 0.05, "alpha", 0, 2)