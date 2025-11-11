import sys
from pathlib import Path

# -------------------------------------------------------------------
# -------------------------- 🚀 用户配置 --------------------------
# -------------------------------------------------------------------

# 1. 您的原始手势文件 (格式: "路径片段 手势名")1
ORIGINAL_FILE = Path("original_gestures.txt")

# 2. 您想生成的输出文件名 (用于新脚本的输入)
OUTPUT_FILE = Path("gesture_map.txt")

# 3. 【重要】手势名到ID的映射
#    (基于您之前提供的列表)
GESTURE_NAME_TO_ID = {
    "open palm": 0,
    "index up": 1,
    "0-Shape": 2,
    "fist": 3,
    "thumb up": 4,
    "thumb down": 5,
    "L-shape": 7,
    "thumb left": 8,
    "thumb right": 9,
    "OK": 10,
    "Close-Pinch": 11,
    "Open-Pinch": 12,
    "heart-single-hand": 13,
    "heart-two-hand": 14
}

# -------------------------------------------------------------------
# -------------------------- 📜 脚本主体 --------------------------
# -------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    标准化名称以便于查找。
    例如: "Open-Pinch" -> "openpinch"
           "thumb up"   -> "thumbup"
    """
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')

def create_reverse_map():
    """
    创建标准化的 { 'normalized_name': id } 映射
    """
    reverse_map = {}
    for name, id in GESTURE_NAME_TO_ID.items():
        normalized = normalize_name(name)
        if normalized in reverse_map:
            print(f"警告: 发现重复的标准名称 '{normalized}'。")
        reverse_map[normalized] = id
    return reverse_map

def process_file():
    """
    读取原始文件，查找ID，并写入新文件。
    """
    if not ORIGINAL_FILE.exists():
        print(f"❌ 错误: 找不到原始手势文件: {ORIGINAL_FILE}")
        print("请确保该文件存在，或修改脚本中的 ORIGINAL_FILE 变量。")
        sys.exit(1)

    # 1. 创建标准化的 "名称 -> ID" 查找表
    name_to_id_map = create_reverse_map()

    print(f"正在读取 '{ORIGINAL_FILE}'...")
    print(f"将写入 '{OUTPUT_FILE}'...")

    success_count = 0
    fail_count = 0

    # 2. 打开两个文件
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        # 写入文件头
        f_out.write("# 自动生成的 gesture_map.txt 文件\n")
        f_out.write("# 格式: [路径片段] [手势名] [类别ID]\n\n")

        # 3. 逐行处理
        for line in f_in:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith("#"):
                f_out.write(line + "\n") # 保留注释
                continue

            # 拆分路径和名称
            parts = line.split(maxsplit=1) # 按第一个空格拆分
            if len(parts) < 2:
                print(f"  [跳过] 格式错误 (未找到手势名): {line}")
                fail_count += 1
                continue
                
            path_key = parts[0]       # 例如 "b/c"
            original_name = parts[1]  # 例如 "OK" 或 "thumb up"
            
            # 4. 查找ID
            normalized_name = normalize_name(original_name)
            class_id = name_to_id_map.get(normalized_name)
            
            if class_id is None:
                print(f"  [失败] 无法匹配手势名: '{original_name}' (标准化后为: '{normalized_name}')")
                fail_count += 1
                continue

            # 5. 写入新行
            new_line = f"{path_key} {original_name} {class_id}\n"
            f_out.write(new_line)
            success_count += 1

    # 6. 打印总结
    print("\n" + "="*30)
    print("🎉 转换完成!")
    print(f"成功转换: {success_count} 行")
    print(f"转换失败 (未匹配到ID): {fail_count} 行")
    print(f"新的映射文件已保存为: {OUTPUT_FILE}")
    print("="*30)

if __name__ == "__main__":
    process_file()
