# 存放模块

import xgi


class tech:
    def __init__(self):
        """
        使用带旋转的 Bron-Kerbosch 算法查找网络中的所有极大团。
        """
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
    
    
    
if __name__ == '__main__':
    pass
