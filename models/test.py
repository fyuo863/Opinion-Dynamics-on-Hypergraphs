import numpy as np
from itertools import combinations
import math

class HigherOrderOpinionDynamics:
    def __init__(self, simplices, gamma, N, max_order=3):
        self.A = self.simplices_to_adjacency(simplices, N)
        self.gamma = gamma
        self.N = N
        self.max_order = max_order
        self.simplices = simplices
        self.hyperedges = self.compute_hyperedges()
        # print("self.hyperedges:", self.hyperedges)

    def simplices_to_adjacency(self, simplices, N):
        vertices = sorted(set(v for simplex in simplices for v in simplex))
        vertex_index = {v: i for i, v in enumerate(vertices)}
        A = np.zeros((N, N), dtype=int)
        
        for simplex in simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    vi, vj = vertex_index.get(simplex[i], -1), vertex_index.get(simplex[j], -1)
                    if vi != -1 and vj != -1:
                        A[vi, vj] = 1
                        A[vj, vi] = 1
        return A

    def compute_hyperedges(self):
        self.hyperedges = {1: [], 2: [], 3: []}  # 支持3阶超边
        visited_pairs = set()  # 记录已访问的二元对

        for simplex in self.simplices:
            if len(simplex) == 2:
                self.hyperedges[1].append(tuple(sorted(simplex)))
            else:
                for pair in combinations(simplex, 2):
                    sorted_pair = tuple(sorted(pair))
                    if sorted_pair not in visited_pairs:
                        self.hyperedges[1].append(sorted_pair)
                        visited_pairs.add(sorted_pair)
                if len(simplex) == 3:
                    self.hyperedges[2].append(tuple(sorted(simplex)))
        self.hyperedges[1] = list(set(self.hyperedges[1]))
        return self.hyperedges
    
    def dx_dt(self, x, w, alpha):
        dx = np.zeros(self.N)

        # 对每个超边的意见累加再进行 tanh(alpha)
        for d, edges in self.hyperedges.items():
            coeff = self.gamma[d - 1] / math.factorial(d)
            for edge in edges:
                total_opinion = sum(x[j] for j in edge)  # 累加超边内所有意见
                for i in edge:
                    dx[i] += coeff * np.tanh(alpha * total_opinion)


        return w + dx

# 示例运行
if __name__ == "__main__":
    N = 10  # 节点数
    S = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (8, 9)]  # 单纯形列表

    gamma = [1.0, 0.5, 0.2]  # 低阶和高阶交互强度
    w = np.random.uniform(0, 0, N)  # 固有意见变化率
    x = np.random.uniform(-1, 1, N)  # 随机初始化意见

    model = HigherOrderOpinionDynamics(S, gamma, N, max_order=3)
    print("self.hyperedges:", model.hyperedges)

    dx = model.dx_dt(x, w, 0.05)
    print("意见变化率 dx/dt:", dx)