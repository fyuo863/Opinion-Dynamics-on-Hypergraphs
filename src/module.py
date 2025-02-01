# 存放模块

import xgi


class tech:
    def __init__(self):
        self.adj_matrix = None
        self.cliques = []
    
    

    def bron_kerbosch_pivot(self, adj_matrix):
        """
        查找网络中的所有极大团。
        :param adj_matrix: 邻接矩阵（二维列表）
        :return: 所有极大团的列表，每个团是一个节点索引的列表
        """
        self.adj_matrix = adj_matrix
        # 初始化
        n = len(self.adj_matrix)  # 节点数量
        self.cliques = []  # 存储所有极大团
        # 初始调用：R 为空，P 为所有节点，X 为空
        self.bron_kerbosch_recursive([], set(range(n)), set())
        return self.cliques

    def bron_kerbosch_recursive(self, r, p, x):
        """
        递归实现带旋转的 Bron-Kerbosch 算法。
        :param r: 当前团
        :param p: 候选节点集合
        :param x: 已处理过的节点集合
        """
        if not p and not x:
            # 如果 P 和 X 都为空，则 R 是一个极大团
            if len(r) > 1:  # 只保留大小大于 1 的团
                self.cliques.append(sorted(r))
            return

        # 选择旋转节点（pivot）：选择 P ∪ X 中具有最多邻居的节点
        pivot = self.select_pivot(p.union(x))
        
        # 遍历 P \ neighbors(pivot)
        for node in list(p.difference(self.get_neighbors(pivot))):
            # 获取节点的邻居
            neighbors = self.get_neighbors(node)
            # 递归调用
            self.bron_kerbosch_recursive(r + [node], 
                                       p.intersection(neighbors), 
                                       x.intersection(neighbors))
            # 更新 P 和 X
            p.remove(node)
            x.add(node)

    def select_pivot(self, nodes):
        """
        优化选择旋转节点
        :param nodes: 候选节点集合
        :return: 选择的旋转节点
        """
        if not nodes:
            return None
        
        # 选择 P 中度数最大的节点
        max_degree = -1
        pivot = None
        
        for node in nodes:
            degree = sum(self.adj_matrix[node])
            if degree > max_degree:
                max_degree = degree
                pivot = node
        return pivot

    def get_neighbors(self, node):
        """
        获取节点的邻居集合。
        :param node: 节点索引
        :return: 邻居集合
        """
        if not hasattr(self, '_neighbor_cache'):
            # 初始化邻居缓存
            self._neighbor_cache = {}
            for n in range(len(self.adj_matrix)):
                self._neighbor_cache[n] = set(i for i, val in enumerate(self.adj_matrix[n]) if val == 1)
        
        return self._neighbor_cache[node]
    
class MaximalCliqueFinder:
    def __init__(self, adj_matrix):
        """
        初始化极大团查找器
        :param adj_matrix: 邻接矩阵（二维列表）
        """
        self.adj = adj_matrix
        self.n = len(adj_matrix)
        self.cliques = []  # 存储所有极大团

    def _get_neighbors(self, v):
        """ 获取顶点v的邻居集合 """
        return set(i for i in range(self.n) if self.adj[v][i] == 1)

    def find_cliques(self):
        """ 主接口：查找并返回所有极大团 """
        self.cliques = []  # 重置结果
        self._bronkkerbosch(set(), set(range(self.n)), set())
        # 过滤掉单节点团（根据需求可调整）
        self.cliques = [c for c in self.cliques if len(c) >= 2]
        return self.cliques

    def _bronkkerbosch(self, R, P, X):
        """ 修正后的Bron-Kerbosch算法实现 """
        if not P and not X:
            self.cliques.append(sorted(R))
            return
        
        # 遍历P的副本以避免修改原始集合
        for v in list(P):
            neighbors_v = self._get_neighbors(v)
            # 仅当v与R中所有顶点相连时才继续（确保R始终是团）
            if all(self.adj[v][u] == 1 for u in R):
                # 生成新的候选集合并递归
                P_new = P & neighbors_v
                X_new = X & neighbors_v
                self._bronkkerbosch(R | {v}, P_new, X_new)
                # 将v从P移到X
                P.remove(v)
                X.add(v)
    
class GraphAnalyzer:
    def __init__(self, adj_matrix, directed=False):
        """
        初始化图分析器
        :param adj_matrix: 邻接矩阵（二维列表）
        :param directed: 是否为有向图（默认为无向图）
        """
        self.adj_matrix = adj_matrix
        self.directed = directed
        self.n = len(adj_matrix)
        self.cliques = []  # 存储所有极大团

    def directed_to_undirected(self):
        """
        将有向图的邻接矩阵转换为无向图的邻接矩阵
        :return: 无向图的邻接矩阵（二维列表）
        """
        undirected_adj_matrix = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if self.adj_matrix[i][j] == 1 or self.adj_matrix[j][i] == 1:
                    undirected_adj_matrix[i][j] = 1
                    undirected_adj_matrix[j][i] = 1
        return undirected_adj_matrix

    def _get_neighbors(self, v, adj_matrix):
        """ 获取顶点v的邻居集合 """
        return set(i for i in range(self.n) if adj_matrix[v][i] == 1)

    def find_maximal_cliques(self):
        """ 查找并返回所有极大团 """
        # 如果是有向图，先转换为无向图
        if self.directed:
            adj_matrix = self.directed_to_undirected()
        else:
            adj_matrix = self.adj_matrix

        # 重置结果
        self.cliques = []
        self._bronkkerbosch(set(), set(range(self.n)), set(), adj_matrix)
        return self.cliques

    def _bronkkerbosch(self, R, P, X, adj_matrix):
        """ Bron-Kerbosch算法实现 """
        if not P and not X:
            self.cliques.append(sorted(R))
            return

        # 选择轴顶点（P ∪ X中度数最大的顶点）
        pivot = max(P.union(X), key=lambda v: len(self._get_neighbors(v, adj_matrix)))
        # 遍历P中不属于轴顶点邻居的顶点
        for v in list(P - self._get_neighbors(pivot, adj_matrix)):
            neighbors_v = self._get_neighbors(v, adj_matrix)
            # 递归探索包含v的情况
            self._bronkkerbosch(R | {v}, P & neighbors_v, X & neighbors_v, adj_matrix)
            # 回溯：将v从P移到X
            P.remove(v)
            X.add(v)


if __name__ == '__main__':
    pass
