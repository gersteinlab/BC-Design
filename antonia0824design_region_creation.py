import os
import json

def create_design_mask_json():
    """
    根据序列长度和指定的位置区域，生成一个布尔值的掩码（mask）JSON文件。
    """
    # --- 1. 定义路径 ---
    base_path = "./data/antonia0824"
    
    # 输入的JSON文件，包含名称和序列
    input_json_path = os.path.join(base_path, "antonia0824.json")
    
    # 输出的目标JSON文件
    output_json_path = os.path.join(base_path, "design_region.json")

    # --- 2. 定义位置数据 (使用1-based索引) ---
    # 适用于文件名包含 '_v2' 的情况
    positions_v2 = [
        (37, 45), (83, 91), (137, 146), (160, 160),
        (171, 179), (193, 210), (260, 260)
    ]
    
    # 适用于其他情况 (v1)
    positions_v1 = [
        (41, 49), (87, 95), (141, 150), (164, 164),
        (175, 183), (197, 214), (264, 264)
    ]

    # --- 3. 加载输入的JSON文件 ---
    print(f"正在读取输入文件: {input_json_path}...")
    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件 '{input_json_path}' 未找到。请检查路径是否正确。")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 '{input_json_path}' 不是一个有效的JSON文件。")
        return

    # 为了方便通过名字快速查找序列，我们将列表转换为字典
    sequences_map = {item['name']: item['seq'] for item in input_data}

    # --- 4. 处理数据并生成布尔值列表 ---
    output_data = {}
    print("正在为每个序列生成布尔值掩码 (mask)...")

    # 遍历我们创建的字典中的每一个条目
    for name, seq in sequences_map.items():
        seq_len = len(seq)
        
        # 创建一个长度为序列长度，且所有值都为False的列表
        mask = [False] * seq_len

        # 根据文件名判断应该使用v1还是v2的位置列表
        positions_to_use = positions_v2 if "_v2" in name else positions_v1

        # 遍历位置区间，并将对应位置设为True
        for start, end in positions_to_use:
            # 将蛋白质序列的1-based索引转换为Python列表的0-based索引
            # 例如，位置(37, 45)对应列表索引应为 36 到 44
            for i in range(start - 1, end):
                if i < seq_len:  # 添加安全检查，防止定义的位置超出实际序列长度
                    mask[i] = True
        
        # 将生成的布尔值列表（mask）存入最终的输出字典
        output_data[name] = mask

        true_count = sum(mask)
        if seq_len > 0:
            print('length: ', seq_len)
            percentage = (true_count / seq_len) * 100
            print(f"  - For sequence '{name}', the percentage of True values is: {percentage:.2f}%")
        else:
            print(f"  - For sequence '{name}', the sequence is empty, cannot calculate percentage.")


    # --- 5. 将结果写入新的JSON文件 ---
    print(f"正在将结果写入: {output_json_path}...")
    with open(output_json_path, 'w') as json_file:
        # 使用 indent=4 参数可以使输出的JSON文件格式化，更易于阅读
        json.dump(output_data, json_file, indent=4)

    print(f"✅ 成功! 文件 '{os.path.basename(output_json_path)}' 已在以下路径创建: {base_path}")

# --- 运行主函数 ---
if __name__ == "__main__":
    create_design_mask_json()