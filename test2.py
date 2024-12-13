class Hypergraph:
    def __init__(self):
        self.hyperedges = []  # 用于存储超边，每个超边是一个集合

    def add_hyperedge(self, nodes):
        """
        添加一条超边
        :param nodes: 一个包含多个节点的列表或集合
        """
        self.hyperedges.append(set(nodes))

    def display_hyperedges(self):
        """
        打印所有超边
        """
        for i, edge in enumerate(self.hyperedges):
            print(f"Hyperedge {i + 1}: {edge}")

# 示例
hypergraph = Hypergraph()

# 添加超边
hypergraph.add_hyperedge([1, 2, 3])  # 超边 {1, 2, 3}
hypergraph.add_hyperedge([3, 4])     # 超边 {3, 4}
hypergraph.add_hyperedge([5, 6, 7, 8])  # 超边 {5, 6, 7, 8}

# 打印超边
hypergraph.display_hyperedges()
