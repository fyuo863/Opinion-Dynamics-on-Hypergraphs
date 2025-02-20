# 存放方法

import xgi



class tech:
    def __init__(self):
        pass

    def find_simplex_with_node(self, simplex, target_node):
        """
        找出包含指定节点的所有超边
        :param simplex: 超图数据，列表形式，每个子列表表示一个超边
        :param target_node: 指定的节点
        :return: 包含指定节点的所有超边
        """
        result = []  # 用于存储结果
        for edge in simplex:  # 遍历每个超边
            if target_node in edge:  # 如果指定节点在当前超边中
                result.append(edge)  # 将当前超边加入结果列表
        return result


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

    def find_maximal_cliques(self, shwo=True):
        """ 查找并返回所有极大团 """
        if shwo:
            self.show = 0
        else:
            self.show = 1
        # 如果是有向图，先转换为无向图
        if self.directed:
            adj_matrix = self.directed_to_undirected()
        else:
            adj_matrix = self.adj_matrix

        # 重置结果
        self.cliques = []
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
            if len(R) > self.show:  # 修改处：只添加长度大于1的团
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
