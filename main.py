import xgi
import matplotlib.pyplot as plt

# 创建一个空超图
H = xgi.Hypergraph()

# 添加节点
H.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])

# 添加超边（每个超边是一个节点的集合）
H.add_edges_from([{1, 2, 3}, {3, 4, 5}, {1, 4}, {6,7,8}])

# 绘制超图
xgi.draw(H)

# 显示图形
plt.show()
