def add_element_with_others(arr, indices, n):
    # 确保索引列表是唯一的（如果需要）
    indices = list(indices)
    
    # 如果索引 n 不在 indices 中，可以选择跳过或处理
    if n not in indices:
        raise ValueError("索引 n 不在给定的索引列表中")
    
    # 计算除 n 以外的其他索引的元素之和
    sum_rest = sum(arr[i] for i in indices if i != n)
    
    # 返回 arr[n] 与所有其他索引元素的和
    return arr[n] + sum_rest

# 示例用法
arr = [i for i in range(100)]  # 假设的长度为 100 的数组
indices = [14, 20, 28]  # 给定的索引列表
n = 14  # 给定的数字索引

result = add_element_with_others(arr, indices, n)
print(f"索引为 {n} 的元素与所有其他索引的元素之和为: {result}")
