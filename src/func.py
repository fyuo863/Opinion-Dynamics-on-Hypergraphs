# 存放功能

import numpy as np
import matplotlib.pyplot as plt
import xgi
import networkx as nx
import os
#from src.module import tech


# 图片保存位置
save_folder = "./plots"
if not os.path.exists(save_folder):
    print(f"创建文件夹 '{save_folder}' 。")
    os.makedirs(save_folder)  # 如果文件夹不存在，则创建
else:
    print(f"文件夹 '{save_folder}' 已存在。")


class func:
    def __init__(self, model):
        self.model = model

    def heatmap(self, lengh, config):
        matrix = np.zeros((lengh, lengh))
        # 横向扫描
        for i in range(lengh):
            print(i)
            config["alpha"] = 0
            for j in range(lengh):
                temp_average = [] 
                for _ in range(10):# 多次运行取平均值
                    self.model.data_in(**config)
                    self.model.simulate_opinion_dynamics()
                    temp = np.abs(self.model.opinions[-1])
                    temp_average.append(np.average(temp))
                print(temp_average)
                matrix[i][j] = np.average(temp_average)
                print(f'K:{config["K"]},alpha:{config["alpha"]},结果{matrix[i][j]}')
                config["alpha"] += 0.2
            config["K"] += 0.2
        plt.imshow(matrix, cmap='gray', vmin=np.min(matrix), vmax=np.max(matrix), origin='lower')  # 使用灰度图，值越低越暗，值越高越亮
        plt.colorbar()  # 添加颜色条
        plt.show()
        # return matrix
    
    def opinions_draw(self, config):
        self.model.data_in(**config)
        self.model.simulate_opinion_dynamics()
        #self.model.draw()
        plt.figure(figsize=(10, 6))
        for i in range(self.model.N):
            plt.plot(range(self.model.T), self.model.opinions[:, i], alpha=0.5)  # 绘制每个代理的意见随时间变化
        plt.xlabel('时间')
        plt.ylabel('意见')
        plt.title(f'N={self.model.N},K={self.model.K},alpha={self.model.alpha},beta={self.model.beta}')
        self.save()
        plt.show()
        

    def network_print(self, A):
        # 创建图对象
        G = nx.from_numpy_array(A)
        # 绘制网络图
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G)  # 布局算法
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=5, font_size=8)
        # 显示图形
        plt.show()

    def simplex_print(self, cliques):
        H = xgi.Hypergraph()
        H.add_edges_from(cliques)
        # 使用 barycenter_spring_layout 布局算法计算节点位置，布局基于春力模型，seed用于固定布局结果
        pos = xgi.barycenter_spring_layout(H, seed=1)

        # 创建一个6x2.5英寸的图形和坐标轴
        fig, ax = plt.subplots(figsize=(6, 2.5))

        # 绘制超图H
        ax, collections = xgi.draw(
            H,  # 超图H
            pos=pos,  # 节点的布局位置
            node_fc=H.nodes.degree,  # 节点的颜色映射：节点度数（连接的超边数）
            edge_fc=H.edges.size,  # 边的颜色映射：超边大小（连接的节点数）
            edge_fc_cmap="viridis",  # 边的颜色映射使用viridis配色方案
            node_fc_cmap="mako_r",  # 节点的颜色映射使用反转的Mako配色方案
        )

        # 从collections中提取节点颜色集合、边颜色集合（中间部分忽略）
        node_col, _, edge_col = collections

        # 为节点度数的颜色映射添加颜色条，并标注为"Node degree"
        plt.colorbar(node_col, label="Node degree")

        # 为超边大小的颜色映射添加颜色条，并标注为"Edge size"
        plt.colorbar(edge_col, label="Edge size")

        # 显示绘制的图形
        plt.show()


    def save(self):
        global save_folder
        # 保存图像
        file_prefix = "Fig"  # 文件名前缀
        file_extension = ".png"  # 文件扩展名
        
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
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")

if __name__ == '__main__':
    pass