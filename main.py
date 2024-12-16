# 每个节点被赋予活动性ai，ai由分布函数F(a)~a^(-gamma)获得
# 在每个时间步，所有节点都有ai的概率被激活，激活时会创建一个(s-1)的单纯形(换成超边)
# (暂定)超边连接的个体为一个小组，组间意见可能分为中立，激化和极化
# 下一时间步，现有的完全子图被清空，重新开始过程

import random
import numpy as np


time_step = 10
num_individuals = 1000# 个体数
a = 0.2# 激活概率(改为activities)
alpha = 0.05
beta = 2.0
gamma = 2.1
m = 10# 尝试连接的节点数

class Hypergraph:
    def __init__(self):
        self.hyperedges = []  # 用于存储超边，每个超边是一个集合

    def add_hyperedge(self, nodes):
        """
        添加一条超边
        :param nodes: 一个包含多个节点的列表或集合
        """
        self.hyperedges.append(set(nodes))

    def del_some_hyperedges(self, index):
        """
        删除指定超边
        示例[1,2]
        """
        self.hyperedges = [s for i, s in enumerate(self.hyperedges) if i not in index]

    
    def del_all_hyperedges(self):
        """
        删除所有超边
        """
        self.hyperedges = []

    def display_hyperedges(self):
        """
        打印所有超边
        """
        for i, edge in enumerate(self.hyperedges):
            print(f"Hyperedge {i + 1}: {edge}")

# def activity_get():# 待完善
#     """
#     获取节点的活动性
#     """
#     # 生成活动性 a_i，分布满足 a^(-gamma)
#     a_min = 0.01  # 避免 a = 0
#     a_max = 1.0
#     rand_value = np.random.uniform(0, 1, size=num_individuals)
#     x_values = np.exp(np.log(rand_value) / (2 * np.log2(10))) - 1
#     a_values = x_values
#     print(f"rand_value{rand_value}")
#     return a_values

def activity_get():# 待完善
    """
    获取节点的活动性
    """
    # 生成活动性 a_i，分布满足 a^(-gamma)
    a_min = 0.01  # 避免 a = 0
    a_max = 1.0
    rand_value = np.random.uniform(0, 1, size=num_individuals)
    x_values = rand_value ** (- 1 / gamma) - 1
    a_values = x_values
    print(f"rand_value{rand_value}")
    return a_values

# def homophily_get(opinions, selected_agent):# 计算同质性
#         # 计算分母denominator
#         denominator = numerator = 0
#         values = []
#         for item in range(num_individuals):
#             if opinions[selected_agent] - opinions[item] == 0:
#                 print("意见相同")
#                 if item != selected_agent:
#                     print("不是自己：赋1")
#                 else:
#                     print("是自己：赋0")
#             if item != selected_agent:
#                 denominator += abs(opinions[selected_agent] - opinions[item]) ** (-beta)
#         # 计算分子numerator
#         for item in range(num_individuals): 
#             # print(f"长度{len(values)}")
#             if item != selected_agent:
#                 values.append(abs(opinions[selected_agent] - opinions[item]) ** (-beta) / denominator)
#             else:
#                 values.append(0)
#         return values

def homophily_get(opinions, node_index):# 计算同质性
    """
    计算给定节点与其他节点之间的同质性.
    
    :param opinions: 一个数组,表示所有节点的意见(x_i)
    :param beta: 指数参数（β）
    :param node_index: 指定的节点索引
    :return: 同质性数组 p_ij
    """
    probabilities = np.zeros(num_individuals)  # 初始化同质性数组
    
    # 计算分母
    denominator = 0
    for j in range(num_individuals):
        if node_index != j:
            denominator += abs(opinions[node_index] - opinions[j]) ** -beta
    
    # 计算每个节点的同质性
    for j in range(num_individuals):
        if node_index != j:
            numerator = abs(opinions[node_index] - opinions[j]) ** -beta
            probabilities[j] = numerator / (denominator + 1e-10)  # 避免分母为0
    
    return probabilities

if __name__ == '__main__':
    hypergraph = Hypergraph()# 实例化
    opinions = np.zeros((num_individuals, time_step))
    # 初始化0时刻意见
    opinions[:, 0] = np.random.uniform(-1, 1, size=num_individuals)
    print(F"初始意见{opinions[:, 0]}")



    for tick in range(1, time_step):

        if tick > 1:# 测试
            break

        # 激活节点
        print(f"当前tick{tick}")
        for item in range(num_individuals):
            print(f"当前节点{item}")
            if random.uniform(0, 1) <= a:# a待替换
                #激活当前节点，当前节点选择节点进行连接(根据同质性)
                print(f"当前节点{item}活跃")
                #获取同质性
                homogeneity = homophily_get(opinions[:, tick - 1], item)
                #根据同质性选择连接的节点(1.直接选择同质性最高的m个节点进行连接。2.依据同质性随机选择m个节点进行连接)
                #1.
                # m_agents = np.argsort(homogeneity)[-m:].tolist()# 索引
                # m_values = homogeneity[m_agents]# 值
                # print(f"准备连接的节点{m_agents}")
                # #尝试连接这m个节点
                # selected_agents = []
                # for value in m_agents:
                #     if random.uniform(0, 1) <= homogeneity[value]:
                #         selected_agents.append(value)
                # print(f"尝试连接的节点：{m_agents}，同质性{m_values}，连接成功的节点：{selected_agents}")
                #2.
                m_agents = []
                m_values = []
                for i in range(m):# 重复选择直至m
                    while 1:
                        rand_flo = random.uniform(0, 1)
                        rand_int = random.randint(0, num_individuals-1)
                        if rand_flo <= homogeneity[rand_int]:
                            m_agents.append(int(rand_int))
                            m_values.append(homogeneity[int(rand_int)])
                            break
                print(f"准备连接的节点{m_agents}")
                #尝试连接这m个节点
                selected_agents = []
                for value in m_agents:
                    if random.uniform(0, 1) <= homogeneity[value]:
                        selected_agents.append(value)
                print(f"尝试连接的节点：{m_agents}，同质性{m_values}，连接成功的节点：{set(selected_agents)}")
                # 将节点用超边连接
                if selected_agents != []:
                    temp = list(set(selected_agents))
                    print(f"temp{temp},item{item}")
                    temp.append(item)
                    hypergraph.add_hyperedge(temp)
                print(f"selected_agents3:{selected_agents},temp:{temp},item:{item}")
        # 打印超边
        print("打印")
        hypergraph.display_hyperedges()
        # 意见传播
        print("占位符")
        for item in hypergraph.hyperedges:
            print(item)
        

    # test = activity_get()
    # print(f"{test}")
    # print(f"最小值{min(test)}，最大值{max(test)}")
                

        
                

                
