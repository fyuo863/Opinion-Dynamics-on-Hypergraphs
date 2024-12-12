# 每个节点被赋予活动性ai，ai由分布函数F(a)~a^(-gamma)获得
# 在每个时间步，所有节点都有ai的概率被激活，激活时会创建一个(s-1)的单纯形(换成超边)
# (暂定)超边连接的个体为一个小组，组间意见可能分为中立，激化和极化
# 下一时间步，现有的完全子图被清空，重新开始过程

import random
import numpy as np


time_step = 10
num_individuals = 10
a = 0.2
alpha = 3.0
beta = 2


def homophily_get(opinions, selected_agent):# 计算同质性
        # 计算分母denominator
        denominator = numerator = 0
        values = []
        for item in range(num_individuals):
            if item != selected_agent:
                denominator += abs(opinions[selected_agent] - opinions[item]) ** (-beta)

        # 计算分子numerator
        for item in range(num_individuals): 
            # print(f"长度{len(values)}")
            if item != selected_agent:
                values.append(abs(opinions[selected_agent] - opinions[item]) ** (-beta) / denominator)
            else:
                values.append(1)
        return values


if __name__ == '__main__':
    opinions = np.zeros((num_individuals, time_step))
    # 初始化0时刻意见
    opinions[:, 0] = np.random.uniform(-1, 1, size=num_individuals)




    for tick in range(time_step):
        # 激活节点
        for item in range(num_individuals):
            if random.uniform(0, 1) <= a:
                #激活当前节点，当前节点选择节点进行连接(根据同质性)
                #获取同质性
                print("哈哈")
                homobility = homophily_get(opinions[:, 0], item)
                
